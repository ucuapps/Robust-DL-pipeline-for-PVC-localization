import os
import pandas as pd
import numpy as np

import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets
import torchvision.transforms as transforms

from ignite.metrics import Accuracy, Precision, Recall, Fbeta, ConfusionMatrix

from dataset.peak_dataset import GaussianPeakDatasetBinary

from sklearn.model_selection import train_test_split

from models.unets import UNet, UNetTime
from modules.utils import plot_seg, compute_f1, compute_iou, threshold_vector
from losses import WingLoss, AdaptiveWingLoss


class SegmentationBaseLine(pl.LightningModule):

    def __init__(self):
        super(SegmentationBaseLine, self).__init__()

        self.model = UNet(1)

        self.create_loss()
        # self.accuracy = Accuracy()
        # self.precision = Precision(average=True)
        # self.recall = Recall(average=True)

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(F.log_softmax(logits), labels)

    def create_loss(self):
        self.loss = nn.MSELoss()
        # self.loss = nn.L1Loss()
        # self.loss = WingLoss()
        # self.loss = AdaptiveWingLoss()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        logits = logits
        loss = self.loss(logits, y)
        logs = {"Training_Loss": loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        logits = self.forward(x)

        prefix = "val" if dataloader_idx else "train"

        loss = self.loss(logits, y)

        seg_ougput = plot_seg(logits[0].cpu().numpy(), y[0].cpu().numpy(), np.squeeze(x[0].cpu().numpy(),0))
        thresholded_logits = threshold_vector(logits[0], 0.5)
        thresholded_y = threshold_vector(y[0])
        thresholded_output = plot_seg(thresholded_logits, thresholded_y, np.squeeze(x[0].cpu().numpy(),0))
        thresholded_output = transforms.ToTensor()(thresholded_output)

        seg_output = transforms.ToTensor()(seg_ougput)
        self.logger.experiment.add_image("{}_y".format(prefix), seg_output, batch_idx)
        self.logger.experiment.add_image("{}_y_thresholded".format(prefix), thresholded_output, batch_idx)

        cur_f1_weighted, cur_f1_samples = compute_f1(y, logits, threshold=0.5)
        cur_iou_weighted, cur_iou_samples = compute_iou(y, logits, threshold=0.5)

        return {"{}_loss".format(prefix): loss,
                "{}_f1_weighted".format(prefix): cur_f1_weighted,
                "{}_f1_samples".format(prefix): cur_f1_samples,
                "{}_iou_weighted".format(prefix): cur_iou_weighted,
                "{}_iou_samples".format(prefix): cur_iou_samples

                }

    def validation_epoch_end(self, outputs):
        def get_average_metric_value(outputs, dataloaders_names, metric_name):
            metric_values = {}
            for i in range(len(dataloaders_names)):
                average_metric = torch.stack([x["{}_{}".format(dataloaders_names[i], metric_name)] for x in outputs[i]]).mean()
                metric_values["{}_{}".format(dataloaders_names[i], metric_name)] = average_metric
            return metric_values

        losses_logs = get_average_metric_value(outputs, ["train", "val"], "loss")
        f1_weighted_logs = get_average_metric_value(outputs, ["train", "val"], "f1_weighted")
        iou_weighted_logs = get_average_metric_value(outputs, ["train", "val"], "iou_weighted")
        iou_samples_logs = get_average_metric_value(outputs, ["train", "val"], "iou_samples")
        f1_samples_logs = get_average_metric_value(outputs, ["train", "val"], "f1_samples")

        self.logger.experiment.add_scalars("Losses", losses_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("IoU_samples", iou_samples_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("F1_samples", f1_samples_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("F1_weighted", f1_weighted_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("IoU_weighted", iou_weighted_logs, global_step=self.current_epoch)

        return {"progress_bar": {**losses_logs, **f1_weighted_logs, **iou_weighted_logs, **f1_samples_logs, **iou_samples_logs}}

    def prepare_data(self):

        exp_path = '/datasets/ecg/china_challenge/'
        df_name = 'icbeb_peaks.csv'

        df = pd.read_csv(os.path.join(exp_path, df_name))
        df = df[df['Location'] > 1000]

        df = df.sample(frac=1).reset_index(drop=True)
        val_df = df[df['PathToData'].isin(['A09.mat', 'A02.mat'])]
        train_df = df[~df['PathToData'].isin(['A09.mat', 'A02.mat'])]

        # train_df, val_df = self.__train_val_split(df)

        train_df.reset_index(inplace=True)
        val_df.reset_index(inplace=True)

        self.train_dataset = GaussianPeakDatasetBinary(exp_path, train_df, augmentation=True)
        self.eval_train_dataset = GaussianPeakDatasetBinary(exp_path, train_df, augmentation=True)
        self.val_dataset = GaussianPeakDatasetBinary(exp_path, val_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=128, num_workers=12)

    def val_dataloader(self):
        return [
            DataLoader(self.eval_train_dataset, num_workers=8, batch_size=128),
            DataLoader(self.val_dataset, num_workers=8, batch_size=128)
            ]

    # def test_dataloader(self):
    #     return DataLoader(self, mnist_test, batch_size=64)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def __train_val_split(self, df):
        train_data, test_data = train_test_split(df, test_size=0.2,
                                                 random_state=42, stratify=df['Label'])
        return train_data, test_data


if __name__ == "__main__":

    from ignite.metrics import Accuracy, Recall, Fbeta, MetricsLambda

    precision = Precision(average=False)
    recall = Recall(average=False)


    # def Fbeta(r, p, beta):
    #     return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
    #
    #
    # m = MetricsLambda(Fbeta, recall, precision, 1)
    # m =
    # batch_size = 5
    # num_classes = 5
    #
    # y_pred = torch.rand(batch_size, num_classes)
    # y = torch.randint(0, num_classes, size=(batch_size,))
    #
    # print(y_pred)
    # print(y)
    # m.update((y_pred, y))
    # res = m.compute()
    # # print(res)
    # m.reset()
    #
