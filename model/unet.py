import torch
import torch.nn as nn
import torch.nn.functional as F

# Double conv module which consists of two conv layer with each followed by a BN nd ReLU
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.features(inputs)


# Left downstream part for unet
class DownStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownStream, self).__init__()
        self.features = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, inputs):
        return self.features(inputs)


# Right upstream part for unet
class UpStream(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpStream, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2)
        self.double_cov = DoubleConv(in_channels, out_channels)

    def forward(self, inputs, down_features):
        # the margin_x and margin_y is even
        upconv_out = self.up_conv(inputs)
        padding_x = down_features.size()[3] - upconv_out.size()[3]
        padding_y = down_features.size()[2] - upconv_out.size()[2]
        upconv_out = F.pad(upconv_out, (padding_x // 2, padding_x - padding_x // 2, padding_y, padding_y - padding_y // 2))
        x = torch.cat([down_features, upconv_out], dim=1)
        out = self.double_cov(x)
        return out


# Unet
class UNet(nn.Module):
    def __init__(self, channels = 1, classes = 1):
        super(UNet, self).__init__()
        self.down1 = DownStream(channels, 64)
        self.down2 = DownStream(64, 128)
        self.down3 = DownStream(128, 256)
        self.down4 = DownStream(256, 512)
        self.double_conv = DoubleConv(512, 1024)
        self.up1 = UpStream(1024, 512)
        self.up2 = UpStream(512, 256)
        self.up3 = UpStream(256, 128)
        self.up4 = UpStream(128, 64)
        self.conv = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, inputs):
        out1 = self.down1(inputs)
        out2 = self.down2(out1)
        out3 = self.down3(out2)
        out4 = self.down4(out3)
        m_conv = self.double_conv(out4)
        out = self.up1(m_conv, out4)
        out = self.up2(out, out3)
        out = self.up3(out, out2)
        out = self.up4(out, out1)
        out = F.sigmoid(self.conv(out))
        return out







