from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    SpatialPadd
)

from monai.transforms import RandAffineD, Rand3DElasticd, RandAdjustContrastd, RandGaussianNoised, RandGaussianSmoothd#, ComputeSDMd


def get_train_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=args.b_min, 
                b_max=args.b_max,
                clip=True,
            ),
            # 🚀 新增：确保图像尺寸至少达到目标 ROI 尺寸
            SpatialPadd(
                keys=["image", "label"], 
                # 目标尺寸应该设置为 RandCropByPosNegLabeld 中使用的 (args.roi_x, args.roi_y, args.roi_z)
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), 
                mode="constant", # 对于图像使用常数填充（例如0）
            ),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[1],
                prob=args.rand_flipd_prob,
            ),
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[2],
                prob=args.rand_flipd_prob,
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=args.rand_rotate90d_prob,
                max_k=3,
            ),
            
            # 🌟 关键：在这里插入 SDM 计算 🌟
            # 必须在 ToTensord 之前，因为 SDM 计算基于 NumPy
            # 必须在所有随机空间变换之后，以确保 SDM 对应增强后的标签
            # 如果您的标签是单通道类别 (例如 0, 1, 2)，您需要指定 num_classes
            # ComputeSDMd(
            #     label_key="label", 
            #     dist_key="dist_maps", 
            #     num_classes=args.out_channels # 仅当标签是单通道类别时才需要
            # ),
            
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=args.rand_shift_intensityd_prob,
            ),
            
            # # additional
            # RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.9,1.1)),
            # RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.01),
            # RandGaussianSmoothd(keys=["image"], prob=0.2, sigma_x=(0.5,1.0), sigma_y=(0.5,1.0), sigma_z=(0.5,1.0)),
            
            ToTensord(keys=["image", "label"])
        ]
    )


def get_val_transform(args):
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            ToTensord(keys=["image", "label"])
        ]
    )


def get_inf_transform(keys, args):
    if len(keys) == 2:
        # image and label
        mode = ("bilinear", "nearest")
    elif len(keys) == 3:
        # image and mutiple label
        mode = ("bilinear", "nearest", "nearest")
    else:
        # image
        mode = ("bilinear")
        
    return Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            Spacingd(
                keys=keys,
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=mode,
            ),
            ScaleIntensityRanged(
                keys=['image'],
                a_min=args.a_min, 
                a_max=args.a_max,
                b_min=args.b_min, 
                b_max=args.b_max,
                clip=True,
                allow_missing_keys=True
            ),
            AddChanneld(keys=keys),
            ToTensord(keys=keys)
        ]
    )


def get_label_transform(keys=["label"]):
    return Compose(
        LoadImaged(keys=keys)
    )