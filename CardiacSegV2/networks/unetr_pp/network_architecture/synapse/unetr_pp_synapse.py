from torch import nn
from typing import Tuple, Union
from networks.unetr_pp.network_architecture.neural_network import SegmentationNetwork
from networks.unetr_pp.network_architecture.dynunet_block import UnetOutBlock, UnetResBlock
from networks.unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec4_out_size,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec3_out_size,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec2_out_size,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=final_out_size,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


from networks.unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder, UnetrUpBlock, UnetrUpBlockLinear
class UNETR_PP_linear(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec4_out_size,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec3_out_size,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec2_out_size,
        )
        self.decoder2 = UnetrUpBlockLinear( # 使用新的線性模塊
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=(4, 4, 4), # 依然是 4x 上採樣
            norm_name=norm_name,
            # out_size 和 conv_decoder 參數不再需要
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits

class UNETR_PP_linear_on3(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec4_out_size,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec3_out_size,
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=final_out_size,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits

class UNETR_PP_linear_on34(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec4_out_size,
        )
        self.decoder4 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=final_out_size,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits

class UNETR_PP_linear_on345(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder4 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=final_out_size,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits

class UNETR_PP_linear_on2345(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder4 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlockLinear( # 使用新的線性模塊
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=(4, 4, 4), # 依然是 4x 上採樣
            norm_name=norm_name,
            # out_size 和 conv_decoder 參數不再需要
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits

class UNETR_PP_linear_on2345(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder4 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlockLinear( # 使用新的線性模塊
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            upsample_kernel_size=(4, 4, 4), # 依然是 4x 上採樣
            norm_name=norm_name,
            # out_size 和 conv_decoder 參數不再需要
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits


class UNETR_PP_linear_0on3(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            img_size: [96, 96, 96],
            feature_size: int = 12,
            hidden_size: int = 192,
            num_heads: int = 4,
            pos_embed: str = "perceptron",  # TODO: Remove the argument
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv3d,
            do_ds=True,

    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.

        Examples::

            # for single channel input 4-channel output with patch size of (64, 128, 128), feature size of 16, batch
            norm and depths of [3, 3, 3, 3] with output channels [32, 64, 128, 256], 4 heads, and 14 classes with
            deep supervision:
            >>> net = UNETR_PP(in_channels=1, out_channels=14, img_size=(64, 128, 128), feature_size=16, num_heads=4,
            >>>                 norm_name='batch', depths=[3, 3, 3, 3], dims=[32, 64, 128, 256], do_ds=True)
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.patch_size = (4, 4, 4)
        self.feat_size = (
            img_size[0] // self.patch_size[0] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[1] // self.patch_size[1] // 8,  # 8 is the downsampling happened through the four encoders stages
            img_size[2] // self.patch_size[2] // 8,  # 8 is the downsampling happened through the four encoders stages
        )
        self.hidden_size = hidden_size

        from networks.unetr_pp.network_architecture.synapse.model_components import UnetrPPEncoder_linear_on0
        self.unetr_pp_encoder = UnetrPPEncoder_linear_on0(
                                                img_size=img_size,
                                                patch_size=self.patch_size,
                                                dims=dims,
                                                depths=depths,
                                                num_heads=num_heads
                                            )

        # -------- decoder out_size 自動計算（避免手動硬編碼） ----------
        # dec4 對應 feat_size * 2, dec3 -> *4, dec2 -> *8, 最終 -> img_size
        fz, fy, fx = self.feat_size
        dec4_size = (fz * 2, fy * 2, fx * 2)
        dec3_size = (fz * 4, fy * 4, fx * 4)
        dec2_size = (fz * 8, fy * 8, fx * 8)
        # convert to scalar product as original code used (e.g. 8*8*8)
        dec4_out_size = dec4_size[0] * dec4_size[1] * dec4_size[2]
        dec3_out_size = dec3_size[0] * dec3_size[1] * dec3_size[2]
        dec2_out_size = dec2_size[0] * dec2_size[1] * dec2_size[2]
        final_out_size = img_size[0] * img_size[1] * img_size[2]

        self.encoder1 = UnetResBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec4_out_size,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=dec3_out_size,
        )
        self.decoder3 = UnetrUpBlockLinear( # <--- 替換成線性模塊
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            # 這裡的 upsample_kernel_size 必須是 2，因為它是 2x 上採樣
            upsample_kernel_size=2, 
            norm_name=norm_name,
            # 由於 UnetrUpBlockLinear 不使用 out_size，可以移除
            # out_size=dec2_out_size, 
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4, 4),
            norm_name=norm_name,
            out_size=final_out_size,
            conv_decoder=True,
        )
        self.out1 = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out2 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
            self.out3 = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 4, out_channels=out_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def forward(self, x_in):
        x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)

        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)
        
        out = self.decoder2(dec1, convBlock)
        if self.do_ds:
            logits = [self.out1(out), self.out2(dec1), self.out3(dec2)]
        else:
            logits = self.out1(out)

        return logits
