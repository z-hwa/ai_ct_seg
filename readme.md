# AI CUP 2025: CT Image Segmentation of Cardiac Muscle

本專案為 **AI CUP 2025 秋季賽：電腦斷層心臟肌肉影像分割** 競賽之實作。我們在 **Private Leaderboard 取得 0.806351 分，排名第 12 名**。

## 專案亮點
* **解碼器結構優化**：針對 UNETR++ 的解碼器進行改良，設計並整合了**線性上採樣層 (Linear Upsampling Layer)**，提升模型泛化能力並降低過擬合風險。
* **損失函數應用**：採用 **Generalized Dice Focal Loss**，有效解決醫學影像中背景與器官體積嚴重不平衡的問題。
* **SOTA 模型消融實驗**：廣泛評估 3D U-Net, Swin UNETR, SegFormer3D 及 UNETR++ 等模型效能。

## 核心技術實現

### 1. 模型架構改良：UNETR++ + Linear Decoder
研究發現原始 UNETR++ 的深層解碼器（Decoder 3）複雜度可能過高，因此我們於 `model_components.py` 中設計了輕量化模塊 `UnetrUpBlockLinear`：
* **機制**：捨棄傳統卷積堆疊，改採**轉置卷積**配合 **1x1 卷積 (Linear Projection)** 融合 Skip Connection 特徵。
* **效果**：實驗證實，將 Decoder 3 替換為線性層後，模型在 Private 資料集取得了更佳表現，驗證了結構優化對提升泛化能力的有效性。

### 2. 訓練策略與損失函數
* **Generalized Dice Focal Loss**：結合 Dice Loss 對輪廓重疊度的敏感性與 Focal Loss 對難分類樣本的權重加強。
* **學習率排程**：使用 **Warmup Cosine Annealing**，前 50 Epochs 線性增加學習率至 5e-4 穩定初期梯度，隨後進行餘弦衰減至第 1600 Epochs。
* **前處理流程**：包括 RAS 座標系重排、重採樣 (0.7mm x 0.7mm x 1mm) 以及針對心臟軟組織特徵的 HU 值截斷 (-42, 423) 與規範化。

## 實驗結果評估 (Private Leaderboard)

我們在相同基準下比較了多種架構，數據顯示引入 Transformer 機制的模型表現顯著優於純 CNN 架構：

| 模型架構 | 損失函數設定 | Private Score (Dice) |
| :--- | :--- | :--- |
| 3D U-Net | Dice Focal Loss | 0.393 |
| Swin UNETR | Dice Focal Loss | 0.455 |
| SegFormer3D | Generalized Dice Focal Loss | 0.764 |
| UNETR++ (Original) | Generalized Dice Focal Loss | 0.805 |
| **UNETR++ + Linear Decoder 3** | **Gen. Dice Focal + Boundary Loss** | **0.806351** |

## 執行環境
* **作業系統**: WSL2 (Ubuntu 20.04)
* **GPU**: NVIDIA GeForce RTX 3060 (12GB)
* **核心框架**: PyTorch 2.9.0, MONAI 1.2.0

## Hot to run this Repo
### 1. Environment install
#### CUDA
Make sure you use cuda 12.6, and having greater than 12G VRAM.
Otherwise, install corresponding PyTorch version.

#### Software
First, run following command under wsl2, to create conda environment.
```
conda env create -f environment.yml
```

second, install python package

```
pip install -r requirements.txt
```

### 2. Dataset Prepare

To reproduce the result, we need to prepare dataset first.

Please download the contest dataset, and unzip it under folder "dataset".
(If folder not exist, created under "CardiacSegV2".)

#### training phase
For training, you need to put data of patient001~patient0050 under "dataset/chgh/train". And patient0046~patient0050 under "dataset/chgh/val".
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

#### predict phase
For submit contest prediction, you need to prepare dataset for test.
* If folder not exist, created by youself.

Put data of patient0051~patient0100 under "dataset/chgh/pred".

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

### 3. training
Before training, ensure your dataset path is correct.
And modify the workspace_dir in "template/AICUP_3D_U_Net_train_2.ipynb" 
```
workspace_dir = "/root/Document/ai_ct_seg/CardiacSegV2"
``` 
to fit your workspace path.

To train the final model, open "template/AICUP_3D_U_Net_train_2.ipynb"
run the command follow the order of code block.

The result will save under "CardiacSegV2/exps/exps/unetr_pp_linear_on3/chgh/tune_results/exp_roi128_gdl_train45/".

### 4. inference
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

---

### 作者 (TEAM_8100)
* **王冠章 (Guan-Zhang Wang)** - 國立成功大學資訊工程學系
