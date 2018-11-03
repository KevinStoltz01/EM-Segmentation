import torch
from torch import nn


class UNet(nn.Module):
    """
    This class defines a UNet model.

    Args:
        channels(int) = The number of color channels per input image
        features_root(int) = The number of feature maps in the first layer of the network. Subsequent layers will be
        defined in reference to this variable.
        output_channels(int) = the number of prediction maps the model will output; defaults to 1.
    """
    def __init__(self, channels=1, features_root=64, output_channels=1, kernel_size=3, activation="relu"):
        super().__init__()
        channels = channels
        features_root = features_root
        if activation not in ["relu", "elu"]:
            raise ValueError("activation must be either 'relu' or 'elu'")
        self.conv1 = ConvBlock(channels, features_root, kernel_size=kernel_size, activation=activation)
        self.down1 = DownSampleBlock()
        self.conv2 = ConvBlock(features_root, features_root*2)
        self.down2 = DownSampleBlock()
        self.conv3 = ConvBlock(features_root*2, features_root*4)
        self.down3 = DownSampleBlock()
        self.conv4 = ConvBlock(features_root*4, features_root*8, dropout=True)
        self.down4 = DownSampleBlock()
        self.conv5 = ConvBlock(features_root*8, features_root*16, dropout=True)
        self.up6 = UpSampleBlock(features_root*16, features_root*8)
        self.conv6 = ConvBlock(features_root*16, features_root*8)
        self.up7 = UpSampleBlock(features_root*8, features_root*4)
        self.conv7 = ConvBlock(features_root*8, features_root*4)
        self.up8 = UpSampleBlock(features_root*4, features_root*2)
        self.conv8 = ConvBlock(features_root*4, features_root*2)
        self.up9 = UpSampleBlock(features_root*2, features_root)
        self.conv9 = ConvBlock(features_root*2, features_root)
        self.conv10 = OutBlock(features_root, output_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        down1 = self.down1(x1)
        x2 = self.conv2(down1)
        down2 = self.down2(x2)
        x3 = self.conv3(down2)
        down3 = self.down3(x3)
        x4 = self.conv4(down3)
        down4 = self.down4(x4)
        x5 = self.conv5(down4)
        up6 = self.up6(x5, x4)
        x6 = self.conv6(up6)
        up7 = self.up7(x6, x3)
        x7 = self.conv7(up7)
        up8 = self.up8(x7, x2)
        x8 = self.conv8(up8)
        up9 = self.up9(x8, x1)
        x9 = self.conv9(up9)
        x10 = self.conv10(x9)

        return x10


# UNet Parts ----------------------------------------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """
    This class defines the convolution block of a UNet model, and is comprised of 2 sequential padded convolutions.

    Args:
         in_fmaps(int) = The number of channels/feature maps to be convolved
         out_fmaps(int) = The number of feature maps output by the convolution operation
         kernel_size(int or tuple) = The nxn dimensions of the convolutional kernel
         padding(int) = Amount of zero padding to be added to the image prior to convolution
         batch_norm(bool) = Toggle batch_normalization on or off. Defaults to True.
         dropout(bool) = Toggle dropout on or off. Defaults to False.

         Note that batch_norm and dropout defaults must be overridden during instantiation in the UNet class
    """

    def __init__(self, in_fmaps, out_fmaps, kernel_size=3, activation="relu", padding=1, batch_norm=True, dropout=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_fmaps, out_fmaps, kernel_size=kernel_size, padding=padding)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        self.batch_norm = nn.BatchNorm2d(out_fmaps)
        if batch_norm:
            self.batch_norm_bool = True
        if not batch_norm:
            self.batch_norm_bool = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        if self.dropout:
            x = nn.Dropout2d(p=0.5)(x)
        x = self.conv2(x)
        x = self.activation(x)
        if self.dropout:
            x = nn.Dropout2d(p=0.5)(x)
        if self.batch_norm_bool:
            x = self.batch_norm(x)

        return x


class DownSampleBlock(nn.Module):
    """
    This class defines the downsampling operations used in UNet. Users can choose from a max pooling operation or
    a strided convolution. The default setup is configured for max pooling.

    Args:
        mode(str; "max_pool" or "stride_conv") = Defines which of the two possible downsampling operations to use.
        in_fmaps(int) = The number of channels/feature maps to be convolved.
        out_fmaps(int) = The number of feature maps output by the convolution operation.
        kernel_size(None, int, or tuple) = The nxn dimensions of the convolutional kernel. To be overridden only for
        use with strided convolution.
        padding(None or int) = Amount of zero padding to be added to input image when strided convolution is used.
    """
    def __init__(self, mode="max_pool", in_fmaps=None, out_fmaps=None, kernel_size=None, padding=None):
        super().__init__()
        self.mode = mode
        if self.mode == "max_pool":
            self.down = nn.MaxPool2d(2, stride=2)
        if self.mode == "stride_conv":
            self.down = nn.Conv2d(in_fmaps, out_fmaps, kernel_size, padding=padding, stride=2)
            self.activation = nn.ReLU()

    def forward(self, x):
        if self.mode == "max_pool":
            x = self.down(x)
        elif self.mode == "down_conv":
            x = self.down(x)
            x = self.activation(x)

        return x


class UpSampleBlock(nn.Module):
    """
    This class defines the upsampling operations used in UNet. Bilinear upsampling + convolution and transpose
    convolution are supported. The default setting is configured for bilinear upsampling + convolution

    Args:
        in_fmaps(int) = The number of channels/feature maps to be convolved.
        out_fmaps(int) = The number of feature maps output by the convolution operation.
        mode(str; "bilinear" or "conv_transpose") = Defines which of the two possible upsampling operations to use.
        kernel_size(None, int, or tuple) = The nxn dimensions of the convolutional kernel.
        padding(None or int) = Amount of zero padding to be added to input image
    """
    def __init__(self, in_fmaps, out_fmaps, mode="bilinear", kernel_size=3, padding=1):
        super().__init__()
        self.mode = mode
        self.activation = nn.ReLU()
        if self.mode == "bilinear":
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size=kernel_size, padding=padding)
        elif self.mode == "conv_transpose":
            self.up_sample = nn.ConvTranspose2d(in_fmaps, out_fmaps, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        x = self.up_sample(x)
        if self.mode == "bilinear":
            x = self.conv(x)
        x = self.activation(x)
        x = torch.cat([x, skip], dim=1)

        return x


class OutBlock(nn.Module):
    """
    This class defines the UNet prediction procedure. For single object segmentation, a single feature map is
    output with or without a sigmoid activation function, depending on whether or not nn.BCEWithLogitsLoss() is used.
    Multi-object segmentation allows the output of multiple feature maps. These feature maps are to be used in
    conjunction with the nn.CrossEntropyLoss() loss function.
    The default settings are configured for single channel output with a sigmoid loss function for use with
    nn.MSELoss() loss function.

    Args:
        in_fmaps(int) = The number of channels/feature maps to be convolved.
        out_fmaps(int) = The number of feature maps output by the convolution operation.
        mode(str; "bilinear" or "conv_transpose") = Defines which of the two possible upsampling operations to use.
        kernel_size(None, int, or tuple) = The nxn dimensions of the convolutional kernel.
        padding(None or int) = Amount of zero padding to be added to input image.
        mode(str; "single" or "multi") = single channel or multi-channel output declaration.
        cross_entropy(bool) = Used to specify outputs when cross entropy losses are used. Some PyTorch cross entropy
        losses compute an activation function for you, therefor these losses to not require that the outputs of the last
        convolutional layer be passed through an activation function before being evaluated by the loss function.
    """
    def __init__(self, in_fmaps, out_fmaps, kernel_size=1, padding=0, mode="single", cross_entropy=False):
        super().__init__()
        self.mode = mode
        self.cross_entropy = cross_entropy
        self.conv = nn.Conv2d(in_fmaps, out_fmaps, kernel_size, padding=padding)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        if self.mode == "single":
            if self.cross_entropy:
                pass
            if not self.cross_entropy:
                x = self.activation(x)
        elif self.mode == "multi":
            pass

        return x
