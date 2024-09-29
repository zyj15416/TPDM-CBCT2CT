import os
import ants
import numpy as np
import tqdm
from skimage import io
import SimpleITK as sitk

if __name__ == '__main__':
    path_in = '/media/ubuntu/DATA_ZYJ/zhirui/6/diffusion_cbct_ct/pelvis'
    path_out = '../data_v1'
    for file in tqdm.tqdm(os.listdir(path_in)):
        try:
            cbct = sitk.GetArrayFromImage(sitk.ReadImage(path_in + '/' + file + '/cbct.nii.gz'))
            ct = sitk.GetArrayFromImage(sitk.ReadImage(path_in + '/' + file + '/ct.nii.gz'))
            mask = sitk.GetArrayFromImage(sitk.ReadImage(path_in + '/' + file + '/mask.nii.gz'))
        except:
            continue
        if not os.path.exists(path_out + '/' + file):
            os.makedirs(path_out + '/' + file)
            os.mkdir(path_out + '/' + file + '/cbct')
            os.mkdir(path_out + '/' + file + '/ct')
            os.mkdir(path_out + '/' + file + '/mask')
        else:
            continue
        assert cbct.shape == ct.shape
        cbct_ = ants.from_numpy(cbct)
        ct_ = ants.from_numpy(ct)
        mask_ = ants.from_numpy(mask)
        registration = ants.registration(ct_, cbct_, 'SyN')
        aligned = registration['warpedmovout']
        cbct = aligned.numpy()
        registration1 = ants.registration(ct_, mask_, 'SyN')
        aligned = registration1['warpedmovout']
        mask = aligned.numpy()
        for i in range(len(cbct)):
            np.save(path_out + '/' + file + '/ct/' + str(i) + '.npy', ct[i])
            np.save(path_out + '/' + file + '/cbct/' + str(i) + '.npy', cbct[i])
            np.save(path_out + '/' + file + '/mask/' + str(i) + '.npy', mask[i])
