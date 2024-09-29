import SimpleITK
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torch import nn
from Src import models
import os
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils.fit import unnormalize_to_zero_to_one
from utils.fit import Fit
from utils import utils
import data_loader
from Src import config

model_use = config.config()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
    return 60 * math.log10(255.0 / rmse)


def SSIM(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    (grayScore, diff) = ssim(imageA, imageB, full=True)

    return grayScore


def unnorm_cbct(x):
    x = x * 355.04248
    x += (-217.94043)
    return x


def unnorm_ct(x):
    x = x * 492.5332
    x += (-440.2342)
    return x


def normalize(img):
    img -= np.min(img)
    img = img / np.max(img)
    return img


if __name__ == "__main__":
    args = utils.get_parse()
    args.training = False
    args.device = [device]
    resize = False
    show = False
    all_test = True
    test_number = 10
    n = -1
    args.sampling_timesteps = 100

    if n == -1:
        model = models.model_T()
        model_data = torch.load('./weights_v3/weights.pth',
                                map_location=device)
    else:
        config_data = model_use.model_config[n]
        model = config_data['model']
        model_data = torch.load('./weights/10/weights.pth',
                                map_location=device)

    try:
        model.load_state_dict(model_data['model_dict'])
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(model_data['model_dict'])
    model = model.to(device)

    if device == 'cuda':
        model = nn.DataParallel(model)
    # print(model)
    gen = Fit(
        model,
        args,
        None,
        None,
        None,
    )

    *_, test_data = data_loader.get_data_path()
    dataloaders_test = DataLoader(data_loader.Dataset_test(test_data),
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=1)
    print(len(dataloaders_test.dataset))
    mse, mae, s, p = [], [], [], []
    rmse = []
    num = 0
    writer = open('./sample/metric.txt', 'w')
    for iteration, batch in enumerate(dataloaders_test):
        if iteration >= test_number:
            break
        batch = batch[0]
        ct_path = batch.replace('/cbct/', '/ct/')
        ct = Image.fromarray(np.load(ct_path))
        ct = ct.resize((args.image_size, args.image_size), Image.Resampling.BICUBIC)
        ct = np.array(ct, np.float64) - (-440.2342)
        label = ct / 492.5332

        pre, pres_all, cond_img = gen.predict(batch)
        pre = pre[0, 0].cpu().numpy()
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(pre), './sample/pre_ct/' + str(iteration) + '.nii')
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(unnorm_ct(label)),
                             './sample/ct/' + str(iteration) + '.nii')
        SimpleITK.WriteImage(SimpleITK.GetImageFromArray(unnorm_cbct(cond_img.cpu().numpy())),
                             './sample/cbct/' + str(iteration) + '.nii')
        if show:
            plt.figure(figsize=(12, 12))
            plt.subplot(221)
            plt.title('cbct')
            plt.imshow(cond_img[0, 0].cpu().numpy(), 'gray')

            plt.subplot(222)
            plt.title('ct')
            plt.imshow(label[0, 0], 'gray')

            plt.subplot(223)
            plt.title('pre_ct')

            print(pre.max())
            plt.imshow(pre, 'gray')

            plt.show()
        mse.append(mean_squared_error(unnorm_ct(label), pre))
        mae.append(mean_absolute_error(unnorm_ct(label), pre))
        p.append(psnr(unnorm_ct(label), pre))
        nor_label = normalize(label)
        nor_pre_ct = normalize(pre)
        s.append(ssim(nor_label, nor_pre_ct,
                      data_range=nor_label.max() - nor_label.min()))
        rmse.append(np.sqrt(mse[-1]))

        writer.write(str(iteration) + ': \n')
        writer.write('mse: ' + str(mse[-1]))
        writer.write(', mae: ' + str(mae[-1]))
        writer.write(', rmse: ' + str(rmse[-1]))
        writer.write(', ssim: ' + str(s[-1]))
        writer.write(', psnr: ' + str(p[-1]))
        writer.write('\n')
        if all_test:
            continue
        else:
            break

    print('mean mse is : ', np.mean(mse), '±', np.std(mse))
    print('mean rmse is : ', np.mean(rmse), '±', np.std(rmse))
    print('mean mae is : ', np.mean(mae), '±', np.std(mae))
    print('mean ssim is : ', np.mean(s), '±', np.std(s))
    print('mean psnr is : ', np.mean(p), '±', np.std(p))
