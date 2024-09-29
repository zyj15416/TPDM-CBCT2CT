import argparse
import json
import os
import shutil
import numpy as np
import torch.nn
from abc import ABC, abstractmethod

from einops import rearrange
from torch.nn import functional as F
from torch import nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1)) if bias else None

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g + default(self.b, 0)


def get_params_groups(model: torch.nn.Module, weights_decay: float = 1e-5):
    # optimizer 要训练的权重
    parameter_group_vars = {
        'decay': {
            'params': [],
            'weight_decay': weights_decay
        },
        'no_decay': {
            'params': [],
            'weight_decay': 0.
        }
    }
    # 对应权重名称
    parameter_group_names = {
        'decay': {
            'params': [],
            'weight_decay': weights_decay
        },
        'no_decay': {
            'params': [],
            'weight_decay': 0.
        }
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if len(param.shape) == 1 or name.endswith('.bias'):
            group_name = 'no_decay'
        else:
            group_name = 'decay'

        parameter_group_vars[group_name]['params'].append(param)
        parameter_group_names[group_name]['params'].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())


def get_parse():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_use", type=int, default=13)
    parse.add_argument("--loss", type=bool, default=False)
    parse.add_argument("--num_epochs", type=int, default=200)
    parse.add_argument("--batch_size", type=int, default=4)
    parse.add_argument("--val_batch_size", type=int, default=4)

    parse.add_argument("--image_size", type=int, default=256)
    parse.add_argument("--weight_decay", type=float, default=1e-2)
    parse.add_argument("--fp16", type=bool, default=True)

    parse.add_argument("--ema_rate", type=str, default="0.9999")
    parse.add_argument("--device", type=tuple, default=['cuda:0'])
    parse.add_argument("--sampling_timesteps", type=int, default=60)
    # objective
    parse.add_argument("--beta_schedule", type=str, default='cosine', help="'cosine' or 'linear'")
    parse.add_argument("--objective", type=str, default='pred_noise',
                       help="'pred_noise' or 'pred_x0' or 'pred_v'")
    parse.add_argument("--timesteps", type=int, default=1000)
    # sample_steps
    parse.add_argument("--sample_steps", type=int, default=1000)
    # lr_decay_type
    parse.add_argument("--lr_decay_type", type=str, default='cosineAnnWarm')
    parse.add_argument("--training", type=bool, default=True)

    parse.add_argument("--load_paramers", type=bool, default=False)
    parse.add_argument("--convert", type=bool, default=False)
    # save_weights_path
    parse.add_argument("--save_weights_path", type=str,
                       default='./weights_v3/')

    args = parse.parse_args()
    return args


class ScheduleSampler(ABC):

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_timesteps):
        self._weights = np.ones([num_timesteps])

    def weights(self):
        return self._weights


def mean_flat(tensor):
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

