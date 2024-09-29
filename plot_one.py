import SimpleITK
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torch import nn
from Src import models
import os
import math
from PIL import Image
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
    n = 12
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
    num = 0
    input_shape = (256, 256)
    img_path = './data/2PA003/cbct/6.npy'
    pre, pres_all, cond_img = gen.predict(img_path)
    pre = pre[0, 0].cpu().numpy()
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.title('cbct')
    plt.imshow(cond_img[0, 0].cpu().numpy(), 'gray')

    plt.subplot(122)
    plt.title('pre_ct')
    plt.imshow(pre, 'gray')

    plt.show()
