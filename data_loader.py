import os
import torch
from PIL import Image
import numpy as np
from torch.utils import data
import albumentations as A


def get_data_path():
    path_all = './data'
    files = []
    for file in sorted(os.listdir(path_all)):
        for i in range(len(os.listdir(path_all + '/' + file + '/ct'))):
            files.append(path_all + '/' + file + '/cbct/' + str(i) + '.npy')
    train_files, test_files = files, files[:len(files) // 10]
    train_files, val_files = train_files, train_files[:len(train_files) // 10]
    train_files, val_files = train_files, train_files[:len(train_files) // 10]
    return train_files, val_files, test_files


def preprocess_input(img):
    img -= np.min(img)
    img = img / (np.max(img) + 1e-3) * 2
    img -= 1
    # img = (img - np.mean(img)) / (np.std(img) + 1e-3)
    return img


class Dataset(data.Dataset):
    def __init__(self, imgs, shape, transform=False, return_roi=False):
        self.use_transform = transform
        self.roi = return_roi
        self.imgs = imgs
        self.input_shape = shape
        self.transform = A.Compose([
            A.Rotate(limit=35, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(p=.3, alpha=120, sigma=120 * .05,
                               alpha_affine=120 * .03),
            A.ShiftScaleRotate(p=.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])

    def read_data(self, img_path):
        label_path = img_path.replace('/cbct/', '/ct/')
        cbct = Image.fromarray(np.load(img_path))
        ct = Image.fromarray(np.load(label_path))
        cbct = cbct.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)
        ct = ct.resize((self.input_shape[0], self.input_shape[1]), Image.BICUBIC)

        if self.roi:
            roi = np.array(ct, np.float64)
        cbct = np.array(cbct, np.float64) - (-217.94043)
        cbct = cbct / 355.04248
        cbct = cbct[..., None]

        ct = np.array(ct, np.float64) - (-440.2342)
        ct = ct / 492.5332
        ct = ct[..., None]
        # cbct = preprocess_input(np.array(cbct, np.float64))[..., None]
        # ct = preprocess_input(np.array(ct, np.float64))[..., None]
        if self.use_transform:
            transformed = self.transform(image=cbct, mask=ct)
            cbct = transformed['image']
            ct = transformed['mask']

        cbct = np.transpose(cbct, [2, 0, 1])
        ct = np.transpose(ct, [2, 0, 1])

        cbct = torch.from_numpy(cbct).type(torch.FloatTensor)
        ct = torch.from_numpy(ct).type(torch.FloatTensor)
        # print(jpg.shape)
        if self.roi:
            return cbct, ct, roi
        else:
            return cbct, ct

    def __getitem__(self, index):
        if self.roi:
            img_x, img_y, roi_x = self.read_data(self.imgs[index])
            return img_y, img_x, roi_x
        else:
            img_x, img_y = self.read_data(self.imgs[index])

            return img_y, img_x

    def __len__(self):
        return len(self.imgs)


class Dataset_test(data.Dataset):
    def __init__(self, imgs):
        self.imgs = imgs

    def __getitem__(self, index):
        return self.imgs[index]

    def __len__(self):
        return len(self.imgs)

if __name__ == '__main__':
    x, y, z = get_data_path()
    print(x)
    img = np.array(Image.open(x[0]))
    print(img.max())
    print(img)
