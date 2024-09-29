import math
import copy
import os
from functools import partial
from collections import namedtuple
import torch as th
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch
import torch.nn.functional as F
from utils.utils import UniformSampler, mean_flat
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm
from PIL import Image
import numpy as np

INITIAL_LOG_LOSS_SCALE = 20.0
ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'out'])


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


# def normalize_to_neg_one_to_one(img):
#     return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    # return (t + 1) * .5
    return (t * 492.5332) + (-440.2342)


def extract(a, t, x_shape):
    b, *_ = t.shape
    a = a.to(t.device)
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Fit:
    def __init__(
            self,
            model,
            args,
            optimizer,
            dataload,
            dataload_val
    ):
        super().__init__()
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.args = args
        self.optimizer = optimizer
        self.data_load = dataload
        self.data_load_val = dataload_val
        self.model = model
        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.channels = 1
        self.device = args.device[0]
        self.ddim_sampling_eta = 1
        # self.ema = args.ema
        self.image_size = args.image_size
        self.ema_rate = (
            [args.ema_rate]
            if isinstance(args.ema_rate, float)
            else [float(x) for x in args.ema_rate.split(",")]
        )
        self.ema_params = [
            copy.deepcopy(self.master_params) for rate in self.ema_rate
        ]
        self.fp16_scale_growth = 1e-3
        self.schedule_sampler = UniformSampler(args.timesteps)

        self.objective = args.objective

        assert args.objective in {'pred_noise', 'pred_x0',
                                  'pred_v'}, \
            'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if args.beta_schedule == 'linear':
            betas = linear_beta_schedule(args.timesteps)
        elif args.beta_schedule == 'cosine':
            betas = cosine_beta_schedule(args.timesteps)
        else:
            raise ValueError(f'unknown beta schedule {args.beta_schedule}')
        if args.training:
            if args.lr_decay_type == 'cosineAnn':
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=5, eta_min=1e-9)
            elif args.lr_decay_type == 'cosineAnnWarm':
                self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=5, T_mult=1)
            else:
                raise ValueError(f'unknown lr decay type {args.lr_decay_type}')
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.sample_steps = args.sample_steps

        self.sampling_timesteps = default(args.sampling_timesteps,
                                          timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        print(self.is_ddim_sampling)

        self.betas = betas.to(torch.float32)
        self.alphas_cumprod = alphas_cumprod.to(torch.float32)
        self.alphas_cumprod_prev = alphas_cumprod_prev.to(torch.float32)

        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(torch.float32)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(torch.float32)
        self.log_one_minus_alphas_cumprod = torch.log(1. - alphas_cumprod).to(torch.float32)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod).to(torch.float32)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1).to(torch.float32)

        posterior_variance = (betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)).to(torch.float32)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20)).to(torch.float32)
        self.posterior_mean_coef1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)).to(torch.float32)
        self.posterior_mean_coef2 = ((1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)).to(
            torch.float32)
        self.scalar = GradScaler()
        self.save_train_loss = []
        self.save_val_loss = []

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, c, clip_denoised=True):
        preds = self.model_predictions(x, t, c)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = \
            self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t, c, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, c=c,
                                                                          clip_denoised=clip_denoised)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        # print(model_mean.shape)
        # print(noise.shape)
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)
        # print(img.shape)

        x_start = None
        imgs = []
        imgs.append(img)

        for t in tqdm(reversed(range(0, self.sample_steps)),
                      desc='sampling loop time step', total=self.sample_steps):
            img, x_start = self.p_sample(img, t, cond)
            imgs.append(img)
            # print(img.shape)

        img = unnormalize_to_zero_to_one(img)
        imgs.append(img)
        return img, imgs

    @torch.no_grad()
    def ddim_sample(self, shape, cond_img, clip_denoised=True, y=None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = \
            shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, \
            self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        # x_start = None
        seg = None
        imgs = [unnormalize_to_zero_to_one(img)]

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            pred_noise, x_start, out = self.model_predictions(img, time_cond, cond_img,
                                                              clip_x_start=clip_denoised)
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
            imgs.append(unnormalize_to_zero_to_one(img))
        img = unnormalize_to_zero_to_one(img)
        imgs.append(img)
        if y is not None:
            return unnormalize_to_zero_to_one(y) * .5 + \
                   .5 * (img * .3 + .7 * unnormalize_to_zero_to_one(out.cpu().numpy())), imgs
        else:
            return img * .3 + .7 * unnormalize_to_zero_to_one(out.cpu().numpy()), imgs

    @torch.no_grad()
    def predict(self, path_cbct):
        cbct = Image.fromarray(np.load(path_cbct))
        cbct = cbct.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
        cbct = np.array(cbct, np.float64) - (-217.94043)
        cbct = cbct / 355.04248
        cbct = cbct[..., None]
        cbct = np.transpose(cbct, [2, 0, 1])
        cbct = torch.from_numpy(cbct).type(torch.FloatTensor)
        cond_img = cbct.to(self.device)[None]
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)
        y = self.get(path_cbct)
        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else \
            self.ddim_sample
        return *sample_fn((batch_size, channels, image_size, image_size), cond_img, y=y), cond_img

    @torch.no_grad()
    def sample(self, cond_img, y):
        batch_size, device = cond_img.shape[0], self.device
        cond_img = cond_img.to(self.device)

        image_size, channels = self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else \
            self.ddim_sample
        return sample_fn((batch_size, channels, image_size, image_size), cond_img, y=y)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def model_predictions(self, x, t, c, clip_x_start=False):
        model_output, out = self.model(x, t, c)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            pred_noise = pred_noise.to(x.device)
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start).to(x.device)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        return ModelPrediction(pred_noise, x_start, out)

    def get_boundary(self, x):
        laplacian_kernel_target = torch.tensor(
            [-1, -1, -1, -1, 8, -1, -1, -1, -1],
            dtype=torch.float32, device=self.args.device[0]).reshape(1, 1, 3, 3).requires_grad_(False)
        x = x.float()
        x = F.conv2d(x, laplacian_kernel_target, padding=1)
        x = x.clamp(min=0, max=1.0)
        x[x >= 0.1] = 1
        x[x < 0.1] = 0

        return x

    def compute_edge_loss(self, logits, targets):
        bs = logits.size()[0]
        boundary_targets = self.get_boundary(targets)
        boundary_targets = boundary_targets.view(bs, 1, -1)
        logits = F.sigmoid(logits)
        boundary_pre = self.get_boundary(logits)
        boundary_pre = boundary_pre / (boundary_pre + 0.01)
        boundary_pre = boundary_pre.view(bs, 1, -1)
        edge_loss = F.binary_cross_entropy_with_logits(boundary_pre, boundary_targets)

        return edge_loss

    def p_losses(self, x_start, t, cond, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_out, out = self.model(x, t, cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')
        assert model_out.shape[1] == target.shape[1]
        if self.args.loss:
            return mean_flat((model_out - target) ** 2) + 1e-3 * self.compute_edge_loss(model_out, target) + \
                   F.mse_loss(out, x_start) + F.l1_loss(out, x_start)
        return mean_flat((model_out - target) ** 2) + \
               F.mse_loss(out, x_start) + F.l1_loss(out, x_start)

    def fit(self):
        save_loss = 100
        for epoch in range(self.args.num_epochs):
            dt_size = len(self.data_load.dataset)
            dt_size_val = len(self.data_load_val.dataset)
            epoch_loss = 0
            step = 0
            device = self.device
            pbar = tqdm(total=dt_size // self.args.batch_size,
                        desc=f'Epoch {epoch + 1}/{self.args.num_epochs}', postfix=dict,
                        mininterval=0.3)

            for x, y in self.data_load:
                inputs = x.to(device)
                labels = y.to(device)
                b, c, h, w = inputs.shape
                self.optimizer.zero_grad()
                times, weights = self.schedule_sampler.sample(b, device)
                if self.args.fp16:
                    with autocast():

                        loss = self.p_losses(inputs, times, labels)
                        loss = (loss * weights).mean()
                else:
                    loss = self.p_losses(inputs, times, labels)
                    loss = (loss * weights).mean()
                epoch_loss += loss.item()
                self.save_train_loss.append(loss.item())

                if self.args.fp16:
                    self.scalar.scale(loss).backward()
                    # losses.backward()
                    self.scalar.step(self.optimizer)
                    self.scalar.update()
                    # loss_scale = 2 ** INITIAL_LOG_LOSS_SCALE
                    # loss = loss_scale * loss
                else:
                    loss.backward()
                    self.optimize_normal()
                pbar.set_postfix(**{'train_loss': epoch_loss / (step + 1),
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
                step += 1
            pbar.close()
            self.scheduler.step()
            pbar = tqdm(total=dt_size_val // self.args.val_batch_size,
                        desc=f'Val_Epoch {epoch + 1}/{self.args.num_epochs}', postfix=dict,
                        mininterval=0.3)
            epoch_loss_val = 0
            step_val = 0
            for x, y in self.data_load_val:
                inputs = x.to(device)
                labels = y.to(device)
                b, c, h, w = inputs.shape
                times, weights = self.schedule_sampler.sample(b, device)
                with torch.no_grad():
                    loss = self.p_losses(inputs, times, labels)
                    loss = (loss * weights).mean()
                epoch_loss_val += loss.item()
                self.save_val_loss.append(loss.item())

                pbar.set_postfix(**{'val_loss': epoch_loss_val / (step_val + 1),
                                    'lr': get_lr(self.optimizer)})
                pbar.update(1)
                step_val += 1
            pbar.close()
            if save_loss > epoch_loss_val / step_val:
                save_loss = epoch_loss_val / step_val
                torch.save(
                    {
                        'epoch': epoch + 1,
                        'optimizer_dict': self.optimizer.state_dict(),
                        'model_dict': self.model.state_dict(),
                    }, self.args.save_weights_path + 'weights.pth'
                )
        with open('./train_loss.txt', 'w') as f:
            for loss in self.save_train_loss:
                f.write(str(loss))
                f.write('\n')

        with open('./val_loss.txt', 'w') as f:
            for loss in self.save_val_loss:
                f.write(str(loss))
                f.write('\n')

    def _master_params_to_state_dict(self, master_params):
        if self.args.fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def get(self, path):
        if os.path.exists(path.replace('/cbct/', '/ct/')):
            y = Image.fromarray(np.load(path.replace('/cbct/', '/ct/')))
            y = y.resize((self.args.image_size, self.args.image_size), Image.BICUBIC)
            y = np.array(y, np.float64) - (-440.2342)
            y = y / 492.5332
            y = y[..., None]
            y = np.transpose(y, [2, 0, 1])
            y = torch.from_numpy(y).type(torch.FloatTensor)
        else:
            y = None
        return y

    def optimizer_fp16(self):
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            return

        model_grads_to_master_grads(self.model_params, self.master_params)
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        master_params_to_model_params(self.model_params, self.master_params)
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        self.optimizer.step()
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)


def model_grads_to_master_grads(model_params, master_params):
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    model_params = list(model_params)

    for param, master_param in zip(
            model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def update_ema(target_params, source_params, rate=0.99):
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)
