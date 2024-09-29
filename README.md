
# Texture-Preserving Diffusion Model for CBCT-to-CT Synthesis
![flow chart](./figure/flowchart.png)



# Dataset
## You should structure your aligned dataset in the following way:
```
data_path/
  2PA092
    ├── CT
    │   ├── 0.npy
    │   ├── 1.npy
    │   ├── ...
    ├── CBCT
    │   ├── 0.npy
    │   ├── 1.npy
    │   ├── ...
  2PA093
    ├── CT
    │   ├── 0.npy
    │   ├── 1.npy
    │   ├── ...
    ├── CBCT
    │   ├── 0.npy
    │   ├── 1.npy
    │   ├── ...
  ...

```

The dataset links used in this study are as follows:
1. [SynthRAD2023](https://synthrad2023.grand-challenge.org/)
2. [Pelvic-ReferenceData](https://wiki.cancerimagingarchive.net/display/Public/Pelvic-ReferenceData)



Before using the dataset, convert the `nii.gz` data to `npy` format using `reg/nii2npy.py`, and perform registration using `syn`.   


# Train
```
python train.py
```
# Predict
```
python predict.py
```

# Pretrained Models
```
We have released pretrained diffusive generators for cbct->ct in SynthRAD2023 challenge dataset( https://pan.baidu.com/s/13c8BlY6XOIuoyXeHtvNxgw?pwd=au45). You can save these weights in relevant checkpoints folder and perform inference.
```

# Citation
```
If you find this repository useful for your research, please use the following.
{

}
```
