# AI CUP 2025 秋季賽 電腦斷層心臟肌肉影像分割

## 1. Environment install
### CUDA
Make sure you use cuda 12.6, and having greater than 12G VRAM.
Otherwise, install corresponding PyTorch version.

### Software
First, run following command under wsl2, to create conda environment.
```
conda env create -f environment.yml
```

second, install python package

```
pip install -r requirements.txt
```

## 2. Dataset Prepare

To reproduce the result, we need to prepare dataset first.

Please download the contest dataset, and unzip it under folder "dataset".
(If folder not exist, created under "CardiacSegV2".)

### training phase
For training, you need to put data of patient001~patient0050 under "dataset/train". And patient0046~patient0050 under "dataset/val".
* If the folder does not exist, created by yourself.
* Data including image and label file, for example patient0001.nii.gz and patient0001_gt.nii.gz

the final path will look like
```
CardiacSegV2
-dataset
    -train
        -patient0001.nii.gz
        -patient0001_gt.nii.gz
        -patient0002.nii.gz
        -patient0002_gt.nii.gz
        ...
        -patient0050.nii.gz
        -patient0050_gt.nii.gz
    -val
        -patient0046.nii.gz
        -patient0046_gt.nii.gz
        ...
        -patient0050.nii.gz
        -patient0050_gt.nii.gz
```

### predict phase
For submit contest prediction, you need to prepare dataset for test.
* If folder not exist, created by youself.

Put data of patient0051~patient0100 under "dataset/pred".

the final path will look like
```
CardiacSegV2
-dataset
    -pred
        -patient0051.nii.gz
        -patient0052.nii.gz
        ...
        -patient0100.nii.gz
    -train
        -patient0001.nii.gz
        -patient0001_gt.nii.gz
        -patient0002.nii.gz
        -patient0002_gt.nii.gz
        ...
        -patient0050.nii.gz
        -patient0050_gt.nii.gz
    -val
        -patient0046.nii.gz
        -patient0046_gt.nii.gz
        ...
        -patient0050.nii.gz
        -patient0050_gt.nii.gz
```

## 3. training
Before training, ensure your dataset path is correct.
And modify the workspace_dir in "template/AICUP_3D_U_Net_train_2.ipynb" 
```
workspace_dir = "/root/Document/ai_ct_seg/CardiacSegV2"
``` 
to fit your workspace path.

To train the final model, open "template/AICUP_3D_U_Net_train_2.ipynb"
run the command follow the order of code block.

The result will save under "CardiacSegV2/exps/exps/unetr_pp_linear_on3/chgh/tune_results/exp_roi128_gdl_train45/".

## 4. inference
To inference, make sure your pred dataset path is right.

And modify the "workspace_dir" in "template/AICUP_3D_U_Net_inference_2.ipynb" 
```
workspace_dir = '/root/Document/ai_ct_seg/CardiacSegV2'
```
to fit your workspace path.

And modify "model_dir" path to your training path which define in "推論設定" block in "template/AICUP_3D_U_Net_inference_2.ipynb" 
```
model_dir = "/root/Document/ai_ct_seg/CardiacSegV2/exps/exps/unetr_pp_linear_on3/chgh/tune_results/exp_roi128_gdl_train45/main_0ab9c_00000_0_exp=exp_exp_roi128_gdl_train45_2025-11-10_13-42-28/models/" # get_tune_model_dir(root_exp_dir, exp_name)
```

After that, run the code block in order.

You will get each patient's predict result under "infer_dir" and zip file at "infer_dir/../predict.zip".
