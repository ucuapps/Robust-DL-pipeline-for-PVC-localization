import torch
import torch.nn as nn

from models.utils import ConvBlockUnet
from torch.nn import functional as F

from .utils import Conv1dSamePadding

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, 12, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv1d(out_channels, out_channels, 12, padding=1),
        nn.ReLU(inplace=True)
    )


class InceptionBlock(nn.Module):
    """An inception block consists of an (optional) bottleneck, followed
    by 3 conv1d layers. Optionally residual
    """

    def __init__(self, in_channels: int, out_channels: int,
                 residual: bool, stride: int = 1, bottleneck_channels: int = 32,
                 kernel_size: int = 41) -> None:
        super().__init__()

        self.use_bottleneck = bottleneck_channels > 0
        if self.use_bottleneck:
            self.bottleneck = Conv1dSamePadding(in_channels, bottleneck_channels,
                                                kernel_size=1, bias=False)
        kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
        start_channels = bottleneck_channels if self.use_bottleneck else in_channels
        channels = [start_channels] + [out_channels] * 3
        self.conv_layers = nn.Sequential(*[
            Conv1dSamePadding(in_channels=channels[i], out_channels=channels[i + 1],
                              kernel_size=kernel_size_s[i], stride=stride, bias=False)
            for i in range(len(kernel_size_s))
        ])

        self.batchnorm = nn.BatchNorm1d(num_features=channels[-1])
        self.relu = nn.ReLU()

        self.use_residual = residual
        if residual:
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU()
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore
        org_x = x
        if self.use_bottleneck:
            x = self.bottleneck(x)
        x = self.conv_layers(x)

        if self.use_residual:
            x = x + self.residual(org_x)
        return x


class UNetTime(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = InceptionBlock(1, 64,True,1, 1,30)
        self.dconv_down2 = InceptionBlock(64, 128,True,1, 1,30)
        self.dconv_down3 = InceptionBlock(128, 256,True,1, 1,30)
        self.dconv_down4 = InceptionBlock(256, 512,True,1, 1,30)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = InceptionBlock(256 + 512, 256,True,1, 1,30)
        self.dconv_up2 = InceptionBlock(128 + 256, 128,True,1, 1,30)
        self.dconv_up1 = InceptionBlock(128 + 64, 64,True,1, 1,30)

        self.conv_last = nn.Conv1d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = torch.sigmoid(out)
        out = torch.squeeze(out, 1)
        return out

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()

        self.dconv_down1 = ConvBlockUnet(1, 64, 5, 1)
        self.dconv_down2 = ConvBlockUnet(64, 128, 5, 1)
        self.dconv_down3 = ConvBlockUnet(128, 256, 5, 1)
        self.dconv_down4 = ConvBlockUnet(256, 512, 5, 1)

        self.maxpool = nn.MaxPool1d(2)
        self.upsample = nn.Upsample(scale_factor=2)

        self.dconv_up3 = ConvBlockUnet(256 + 512, 256, 5, 1)
        self.dconv_up2 = ConvBlockUnet(128 + 256, 128, 5, 1)
        self.dconv_up1 = ConvBlockUnet(128 + 64, 64, 5, 1)

        self.conv_last = nn.Conv1d(64, n_class, 1)

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        x = self.dconv_down4(x)

        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        out = torch.sigmoid(out)
        out = torch.squeeze(out, 1)
        return out


if __name__ == "__main__":

    import pandas as pd
    import os
    from dataset.peak_dataset import ECGPeaksDataset
    from torch.utils.data import DataLoader

    exp_path = '/datasets/ecg/china_challenge'
    df_name = 'icbeb_peaks.csv'
    df = pd.read_csv(os.path.join(exp_path, df_name))
    df = df.sample(frac=1).reset_index(drop=True)
    train_df = df[~df['PathToData'].isin(
        ['/datasets/ecg/china_challenge/TrainingSet/A09.mat', '/datasets/ecg/china_challenge/TrainingSet/A02.mat'])]
    train_df.reset_index(inplace=True)
    train_dataset = ECGPeaksDataset(exp_path, train_df, augmentation=True)
    dl = DataLoader(train_dataset, batch_size=1, num_workers=12)

    unet = UNet(3)

    for x, y in dl:
        output = unet(x)
        print(output.shape)
        print(output.max())
