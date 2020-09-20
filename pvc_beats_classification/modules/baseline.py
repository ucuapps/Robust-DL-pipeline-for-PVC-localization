# import os
# import numpy as np
# import pandas as pd
# import wfdb
# import random
# import torch
# from torch import nn
# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, random_split
# from torch.nn import functional as F
# from torchvision.datasets import MNIST
# from torchvision import datasets, transforms
#
# from ignite.metrics import Accuracy, Precision, Recall, Fbeta, ConfusionMatrix
#
# from dataset.peak_dataset import ECGPeaksDataset, PQRSTDataset, ECGPeaksDatasetMIT
#
# from sklearn.model_selection import train_test_split
#
# from models.fcn import FCNBaseline
# from models.inception import InceptionModel
# from models.resnet import ResNetBaseline
# from models.rnn_model import BidirectionalLSTM
#
#
# class BaseLine(pl.LightningModule):
#
#     def __init__(self, fold):
#         super(BaseLine, self).__init__()
#
#         self.model = FCNBaseline(1, 3)
#         self.fold = fold
#         self.accuracy = Accuracy()
#         self.precision = Precision(average=True)
#         self.recall = Recall(average=True)
#
#         # self.f_beta = Fbeta(1)
#
#     def forward(self, x):
#         return self.model(x)
#
#     def cross_entropy_loss(self, logits, labels):
#         return F.nll_loss(F.log_softmax(logits), labels)
#
#     def training_step(self, train_batch, batch_idx):
#         x, y = train_batch
#         logits = self.forward(x)
#         loss = self.cross_entropy_loss(logits, y)
#         logs = {"Training_Loss": loss}
#         return {'loss': loss, 'log': logs}
#
#     def validation_step(self, batch, batch_idx, dataloader_idx):
#
#         x, y = batch
#         logits = self.forward(x)
#
#         prefix = "val" if dataloader_idx==1 else "test" if dataloader_idx==2 else "train"
#
#         loss = self.cross_entropy_loss(logits, y)
#
#         softamx_logits = F.softmax(logits)
#         return {"{}_loss".format(prefix): loss,
#                 "{}_y".format(prefix): y,
#                 "{}_pred".format(prefix): softamx_logits}
#
#     def validation_epoch_end(self, outputs):
#         def get_average_metric_value(outputs, dataloaders_names, metric_name):
#             metric_values = {}
#             for i in range(len(dataloaders_names)):
#                 average_metric = torch.stack([x["{}_{}".format(dataloaders_names[i], metric_name)] for x in outputs[i]]).mean()
#                 metric_values["{}_{}".format(dataloaders_names[i], metric_name)] = average_metric
#             return metric_values
#
#         def compute_metrics(softmax_logits, labels):
#             self.accuracy.update((softmax_logits, labels))
#             self.precision.update((softmax_logits, labels))
#             self.recall.update((softmax_logits, labels))
#
#             cur_acc = torch.tensor(self.accuracy.compute())
#             cur_precicion = torch.tensor(self.precision.compute())
#             cur_recall = torch.tensor(self.recall.compute())
#             cur_f1 = torch.tensor(self.Fbeta(cur_recall, cur_precicion, 1))
#
#             self.accuracy.reset()
#             self.precision.reset()
#             self.recall.reset()
#             return cur_acc, cur_precicion, cur_recall, cur_f1
#
#         labels_train = torch.cat([x['train_y'] for x in outputs[0]])
#         logits_train = torch.cat([x['train_pred'] for x in outputs[0]])
#
#         labels_val = torch.cat([x['val_y'] for x in outputs[1]])
#         logits_val = torch.cat([x['val_pred'] for x in outputs[1]])
#
#         labels_test = torch.cat([x['test_y'] for x in outputs[2]])
#         logits_test = torch.cat([x['test_pred'] for x in outputs[2]])
#
#         train_metrics = compute_metrics(logits_train, labels_train)
#         val_metrics = compute_metrics(logits_val, labels_val)
#         test_metrics = compute_metrics(logits_test, labels_test)
#
#         losses_logs = get_average_metric_value(outputs, ["train", "val", "test"], "loss")
#         acc_logs = {"TRAIN_accuracy": train_metrics[0], "VAL_accuracy": val_metrics[0],
#                     "TEST_accuracy": test_metrics[0]}
#         precision_logs = {"TRAIN_precision": train_metrics[1], "VAL_precision": val_metrics[1],
#                           "TEST_precision": test_metrics[1]}
#         recall_logs = {"TRAIN_recall": train_metrics[2], "VAL_recall": val_metrics[2], "TEST_reall": test_metrics[2]}
#         f1_logs = {"TRAIN_F1": train_metrics[3], "VAL_F1": val_metrics[3], "TEST_F1": test_metrics[3]}
#
#         self.logger.experiment.add_scalars("Losses", losses_logs, global_step=self.current_epoch)
#         self.logger.experiment.add_scalars("Accuracy", acc_logs, global_step=self.current_epoch)
#         self.logger.experiment.add_scalars("Precision", precision_logs, global_step=self.current_epoch)
#         self.logger.experiment.add_scalars("Recall", recall_logs, global_step=self.current_epoch)
#         self.logger.experiment.add_scalars("F1", f1_logs, global_step=self.current_epoch)
#         return {"progress_bar": {**losses_logs, **acc_logs, **precision_logs, **recall_logs}}
#
#     def filter_dataset(self, path_to_df, dataset_name):
#         df = pd.read_csv(path_to_df)
#         df = df[(df['Label'] == 'N') | (df['Label'] == 'PVC')]
#         df = df[(df['Location'] > 201) & (df['Dataset'] == dataset_name)].reset_index(drop=True)
#
#         if dataset_name == 'physionet.org/files/svdb/1.0.0/':
#             df = df[(df['Location'] + 200) < 230400].reset_index(drop=True)  # mit-svdb
#         else:
#             df = df[(df['Location']+200) < 650000].reset_index(drop=True) #mit-bih
#         return df
#
#     def train_val_test_split(self, dataset, cv_breaks,  k_fold_idx, dataset_name):
#         dataset['PathToData'] = dataset['PathToData'].apply(int)
#         cv_breaks['PathToData'] = cv_breaks['PathToData'].apply(int)
#         cv_breaks['Fold'] = cv_breaks['Fold'].apply(int)
#         cv_breaks = cv_breaks[cv_breaks['Dataset'] == dataset_name]
#
#         test_cases = cv_breaks['PathToData'][cv_breaks['Fold'] == -1][:-3]
#         additional_train = cv_breaks['PathToData'][cv_breaks['Fold'] == -1][-3:]
#         test_df = dataset[dataset['PathToData'].isin(test_cases)]
#         val_df = dataset[dataset['PathToData'].isin(cv_breaks['PathToData'][cv_breaks['Fold'] == k_fold_idx])]
#
#         folds = list(pd.unique(cv_breaks['Fold']))
#         folds.remove(-1)
#         folds.remove(k_fold_idx)
#         train_df = dataset[dataset['PathToData'].isin(cv_breaks['PathToData'][cv_breaks['Fold'].isin(folds)])]
#         add_tr_df = dataset[dataset['PathToData'].isin(additional_train)]
#         train_df = pd.concat([train_df, add_tr_df])
#         train_df = pd.concat([train_df[train_df['Label'] == 'N'].sample(5000),
#                               train_df[train_df['Label'] == 'PVC'],
#                               ]).sample(frac=1).reset_index(drop=True)
#
#         val_df = pd.concat([val_df[val_df['Label'] == 'N'].sample(val_df[val_df['Label'] == 'PVC'].shape[0]),
#                               val_df[val_df['Label'] == 'PVC'],
#                               ]).sample(frac=1).reset_index(drop=True)
#
#         test_df = pd.concat([test_df[test_df['Label'] == 'N'].sample(test_df[test_df['Label'] == 'PVC'].shape[0]),
#                               test_df[test_df['Label'] == 'PVC'],
#                               ]).sample(frac=1).reset_index(drop=True)
#
#         train_df.reset_index(inplace=True)
#         val_df.reset_index(inplace=True)
#         test_df.reset_index(inplace=True)
#
#
#         print(pd.value_counts(train_df['Label']))
#         print(pd.value_counts(val_df['Label']))
#         print(pd.value_counts(test_df['Label']))
#
#         return train_df, val_df, test_df
#
#     def prepare_data(self):
#         df_name = '/datasets/extra_space2/ikachko/ecg/mitdb_svdb_resampled.csv'
#         exp_path = '/datasets/extra_space2/ikachko/ecg/'
#         cross_dataset = '/datasets/extra_space2/ikachko/ecg/cross_val_mitbih_svdb_v2.csv'
#
#         # dataset_name = 'physionet.org/files/svdb/1.0.0/'
#         dataset_name = 'physionet.org/files/mitdb/1.0.0/'
#         random.seed(42)
#
#         cross_dataset = pd.read_csv(cross_dataset)
#
#         df = self.filter_dataset(df_name, dataset_name)
#         train_df, val_df, test_df = self.train_val_test_split(df, cross_dataset, self.fold, dataset_name)
#
#         # df_name = '/datasets/extra_space2/ikachko/ecg/china_challenge/icbeb_peaks.csv'
#         # df = pd.read_csv(df_name)
#
#
#         # self.train_dataset = ECGPeaksDataset(exp_path, train_df, augmentation=False)
#         # self.eval_train_dataset = ECGPeaksDataset(exp_path, train_df)
#         # self.val_dataset = ECGPeaksDataset('/datasets/extra_space2/ikachko/ecg/china_challenge', val_df)
#
#         print(train_df.shape, val_df.shape, test_df.shape)
#         self.train_dataset = ECGPeaksDatasetMIT(exp_path, train_df, augmentation=False)
#         self.eval_train_dataset = ECGPeaksDatasetMIT(exp_path, train_df)
#         self.val_dataset = ECGPeaksDatasetMIT(exp_path, val_df)
#         self.test_dataset = ECGPeaksDatasetMIT(exp_path, test_df)
#
#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=128, num_workers=12)
#
#     def val_dataloader(self):
#         return [
#             DataLoader(self.eval_train_dataset, num_workers=8, batch_size=128),
#             DataLoader(self.val_dataset, num_workers=8, batch_size=128),
#             DataLoader(self.test_dataset, num_workers=8, batch_size=128)
#             ]
#
#     # def test_dataloader(self):
#     #     return DataLoader(self, mnist_test, batch_size=64)
#
#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
#         return optimizer
#
#     def Fbeta(self, r, p, beta):
#         return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
#
#     def __train_val_split(self, df):
#         train_data, test_data = train_test_split(df, test_size=0.2,
#                                                  random_state=42, stratify=df['Label'])
#         return train_data, test_data
#
#
# class InceptionTime(BaseLine):
#
#     def __init__(self, fold):
#         super(InceptionTime, self).__init__(fold)
#
#         self.model = InceptionModel(
#             # num_blocks=5,
#             num_blocks=3,
#             in_channels=1,
#             out_channels=100,
#             bottleneck_channels=1,
#             kernel_sizes=[10, 20, 200],
#             use_residuals=True,
#             num_pred_classes=2
#         )
#
#
# class ResNet(BaseLine):
#
#     def __init__(self, fold):
#         super(ResNet, self).__init__(fold)
#
#         self.model = ResNetBaseline(
#             in_channels=1,
#             num_pred_classes=2
#         )
#
#
# class RNNModel(BaseLine):
#     def __init__(self, fold):
#         super(RNNModel, self).__init__(fold)
#
#         self.model = BidirectionalLSTM(
#             input_size=300,
#             output_size=2,
#             hidden_dim=100,
#             n_layers=3,
#             drop_prob=0.5
#         )
#
#
# if __name__ == "__main__":
#
#     from ignite.metrics import Accuracy, Recall, Fbeta, MetricsLambda
#
#     model = RNNModel()
#
#     model.prepare_data()
#
#     tr = model.train_dataset.df
#     val = model.val_dataset.df
#
#     # print(pd.value_counts(tr['Label']))
#     # print(pd.value_counts(val['Label']))
#
#     # def Fbeta(r, p, beta):
#     #     return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()
#     #
#     #
#     # m = MetricsLambda(Fbeta, recall, precision, 1)
#     # m =
#     # batch_size = 5
#     # num_classes = 5
#     #
#     # y_pred = torch.rand(batch_size, num_classes)
#     # y = torch.randint(0, num_classes, size=(batch_size,))
#     #
#     # print(y_pred)
#     # print(y)
#     # m.update((y_pred, y))
#     # res = m.compute()
#     # # print(res)
#     # m.reset()
#     #
#
#
#     # def prepare_data(self):
#     #
#     #     exp_path = '/datasets/ecg/china_challenge'
#     #
#     #     # df_name = 'icbeb_peaks.csv'
#     #     # df_name = 'unet_pqrst_labeled.csv'
#     #     # df_name = 'unet_rpeak_labeled.csv'
#     #     # df = pd.read_csv(os.path.join(exp_path, df_name))
#     #
#     #     # df_name = '/datasets/extra_space2/ikachko/ecg/PVC_mit_bih.csv'
#     #     # df_name = '/datasets/extra_space2/ikachko/ecg/PVC_mit_svdb.csv'
#     #     df_name = '/datasets/extra_space2/ikachko/ecg/mitdb_svdb_resampled.csv'
#     #     exp_path = '/datasets/extra_space2/ikachko/ecg/'
#     #     random.seed(42)
#     #     df = pd.read_csv(df_name)
#     #     df = df[df['Dataset'] == 'physionet.org/files/mitdb/1.0.0/']
#     #     df = df[df['Location'] > 201].reset_index(drop=True)
#     #     df = df[(df['Location']+200) < 650000].reset_index(drop=True) #mit-bih
#     #     # df = df[(df['Location']+200) < 230400].reset_index(drop=True) # mit-svdb
#     #
#     #     # val_samples = random.choices(pd.unique(df[df['Label'] == 'PVC']['PathToData']),
#     #     #                    k=int(len(pd.unique(df[df['Label'] == 'PVC']['PathToData'])) * 0.2))
#     #     # val_samples = df[df['PathToData'].isin(val_samples)]
#     #     train_samples = df#df[~df['PathToData'].isin(val_samples)]
#     #
#     #     # mit-svdb
#     #     # train_df = pd.concat([train_samples[train_samples['Label'] == 'N'].sample(10000),
#     #     #                 train_samples[train_samples['Label'] == 'PVC'],
#     #     #                 ]).sample(frac=1).reset_index(drop=True)
#     #     # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(2000),
#     #     #                 val_samples[val_samples['Label'] == 'PVC'],
#     #     #                 ]).reset_index(drop=True)
#     #
#     #     train_df = pd.concat([train_samples[train_samples['Label'] == 'N'].sample(10000),
#     #                           train_samples[train_samples['Label'] == 'PVC'],
#     #                           ]).sample(frac=1).reset_index(drop=True)
#     #
#     #     # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(500),
#     #     #                     val_samples[val_samples['Label'] == 'PVC'],
#     #     #                     ]).reset_index(drop=True)
#     #
#     #     df_name = '/datasets/extra_space2/ikachko/ecg/china_challenge/icbeb_peaks.csv'
#     #     df = pd.read_csv(df_name)
#     #
#     #
#     #     # df = df[df['Dataset'] == 'physionet.org/files/svdb/1.0.0/']
#     #     # df = df[df['Location'] > 201].reset_index(drop=True)
#     #     # df = df[(df['Location']+200) < 230400].reset_index(drop=True) # mit-svdb
#     #     #
#     #     # val_df = pd.concat([df[df['Label'] == 'N'].sample(12000),
#     #     #                     df[df['Label'] == 'PVC'],
#     #     #                     ]).reset_index(drop=True)
#     #
#     #     val_df = pd.concat([df[df['Label'] == 'Normal'].sample(20000),
#     #                         df[df['Label'] == 'PVC'],
#     #                         ]).reset_index(drop=True)
#     #
#     #     # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(12000),
#     #     #                     val_samples[val_samples['Label'] == 'PVC'],
#     #     #                     ]).reset_index(drop=True)
#     #
#     #     # val_df = pd.read_csv('/datasets/extra_space2/ikachko/ecg/PVC_mit_svdb.csv')
#     #     # val_df = val_df[(val_df['Location']+200) < 230400].reset_index(drop=True) # mit-svdb
#     #     # val_df = val_df[val_df['Location'] > 201].reset_index(drop=True)
#     #     #
#     #     # val_df = pd.concat([val_df[val_df['Label'] == 'N'].sample(12000),
#     #     #                 val_df[val_df['Label'] == 'PVC'],
#     #     #                 ]).reset_index(drop=True)
#     #
#     #     # val_samples = df[df['PathToData'].isin(['A09.mat', 'A02.mat'])]
#     #     # train_samples = df[~df['PathToData'].isin(['A09.mat', 'A02.mat'])]
#     #     #
#     #     # train_df = pd.concat([train_samples[train_samples['Label'] == 'Normal'].sample(30000),
#     #     #                 train_samples[train_samples['Label'] == 'PVC'],
#     #     #                 ]).sample(frac=1).reset_index(drop=True)
#     #     # val_df = pd.concat([val_samples[val_samples['Label'] == 'Normal'].sample(5000),
#     #     #                 val_samples[val_samples['Label'] == 'PVC'],
#     #     #                 ]).reset_index(drop=True)
#     #
#     #
#     #     # train_df, val_df = self.__train_val_split(df)
#     #     # train_df.reset_index(inplace=True)
#     #     # val_df.reset_index(inplace=True)
#     #
#     #     # self.train_dataset = ECGPeaksDataset(exp_path, train_df, augmentation=False)
#     #     # self.eval_train_dataset = ECGPeaksDataset(exp_path, train_df)
#     #     self.val_dataset = ECGPeaksDataset('/datasets/extra_space2/ikachko/ecg/china_challenge', val_df)
#     #     self.train_dataset = ECGPeaksDatasetMIT(exp_path, train_df, augmentation=False)
#     #     self.eval_train_dataset = ECGPeaksDatasetMIT(exp_path, train_df)
#     #     # self.val_dataset = ECGPeaksDatasetMIT(exp_path, val_df)
#     #     # self.train_dataset = PQRSTDataset(exp_path, train_df, augmentation=False)
#     #     # self.eval_train_dataset = PQRSTDataset(exp_path, train_df)
#     #     # self.val_dataset = PQRSTDataset(exp_path, val_df)


































import os
import numpy as np
import pandas as pd
import wfdb
import random
import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms

from ignite.metrics import Accuracy, Precision, Recall, Fbeta, ConfusionMatrix

from dataset.peak_dataset import ECGPeaksDataset, PQRSTDataset, ECGPeaksDatasetMIT

from sklearn.model_selection import train_test_split

from models.fcn import FCNBaseline
from models.inception import InceptionModel
from models.resnet import ResNetBaseline
from models.rnn_model import BidirectionalLSTM


class BaseLine(pl.LightningModule):

    def __init__(self):
        super(BaseLine, self).__init__()

        self.model = FCNBaseline(1, 3)

        self.accuracy = Accuracy()
        self.precision = Precision(average=True)
        self.recall = Recall(average=True)

        # self.f_beta = Fbeta(1)

    def forward(self, x):
        return self.model(x)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(F.log_softmax(logits), labels)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        logs = {"Training_Loss": loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx, dataloader_idx):

        x, y = batch
        logits = self.forward(x)

        prefix = "val" if dataloader_idx else "train"

        loss = self.cross_entropy_loss(logits, y)

        softamx_logits = F.softmax(logits)
        return {"{}_loss".format(prefix): loss,
                "{}_y".format(prefix): y,
                "{}_pred".format(prefix): softamx_logits}

    def validation_epoch_end(self, outputs):
        def get_average_metric_value(outputs, dataloaders_names, metric_name):
            metric_values = {}
            for i in range(len(dataloaders_names)):
                average_metric = torch.stack([x["{}_{}".format(dataloaders_names[i], metric_name)] for x in outputs[i]]).mean()
                metric_values["{}_{}".format(dataloaders_names[i], metric_name)] = average_metric
            return metric_values

        def compute_metrics(softmax_logits, labels):
            self.accuracy.update((softmax_logits, labels))
            self.precision.update((softmax_logits, labels))
            self.recall.update((softmax_logits, labels))

            cur_acc = torch.tensor(self.accuracy.compute())
            cur_precicion = torch.tensor(self.precision.compute())
            cur_recall = torch.tensor(self.recall.compute())
            cur_f1 = torch.tensor(self.Fbeta(cur_recall, cur_precicion, 1))

            self.accuracy.reset()
            self.precision.reset()
            self.recall.reset()
            return cur_acc, cur_precicion, cur_recall, cur_f1

        labels_train = torch.cat([x['train_y'] for x in outputs[0]])
        logits_train = torch.cat([x['train_pred'] for x in outputs[0]])

        labels_val = torch.cat([x['val_y'] for x in outputs[1]])
        logits_val = torch.cat([x['val_pred'] for x in outputs[1]])

        train_metrics = compute_metrics(logits_train, labels_train)
        val_metrics = compute_metrics(logits_val, labels_val)

        losses_logs = get_average_metric_value(outputs, ["train", "val"], "loss")
        acc_logs = {"TRAIN_accuracy": train_metrics[0], "VAL_accuracy": val_metrics[0]}
        precision_logs = {"TRAIN_precision": train_metrics[1], "VAL_precision": val_metrics[1]}
        recall_logs = {"TRAIN_recall": train_metrics[2], "VAL_recall": val_metrics[2]}
        f1_logs = {"TRAIN_F1": train_metrics[3], "VAL_F1": val_metrics[3]}

        self.logger.experiment.add_scalars("Losses", losses_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", acc_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Precision", precision_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Recall", recall_logs, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("F1", f1_logs, global_step=self.current_epoch)
        return {"progress_bar": {**losses_logs, **acc_logs, **precision_logs, **recall_logs}}

    def prepare_data(self):

        exp_path = '/datasets/ecg/china_challenge'

        # df_name = 'icbeb_peaks.csv'
        # df_name = 'unet_pqrst_labeled.csv'
        # df_name = 'unet_rpeak_labeled.csv'
        # df = pd.read_csv(os.path.join(exp_path, df_name))

        # df_name = '/datasets/extra_space2/ikachko/ecg/PVC_mit_bih.csv'
        # df_name = '/datasets/extra_space2/ikachko/ecg/PVC_mit_svdb.csv'
        df_name = '/datasets/extra_space2/ikachko/ecg/mitdb_svdb_resampled.csv'
        exp_path = '/datasets/extra_space2/ikachko/ecg/'
        random.seed(42)
        df = pd.read_csv(df_name)
        df = df[df['Dataset'] == 'physionet.org/files/svdb/1.0.0/']
        # df = df[df['Dataset'] == 'physionet.org/files/mitdb/1.0.0/']

        df = df[df['Location'] > 201].reset_index(drop=True)
        # df = df[(df['Location']+201) < 650000].reset_index(drop=True) #mit-bih
        df = df[(df['Location']+200) < 230400].reset_index(drop=True) # mit-svdb
        # df = df[(df['Location']+201) < 650000].reset_index(drop=True) #mit-bih

        # val_samples = random.choices(pd.unique(df[df['Label'] == 'PVC']['PathToData']),
        #                    k=int(len(pd.unique(df[df['Label'] == 'PVC']['PathToData'])) * 0.4))
        #
        # val_samples = df[df['PathToData'].isin(val_samples)]
        # train_samples = df[~df['PathToData'].isin(val_samples)]

        train_samples = df
        # mit-svdb
        train_df = pd.concat([train_samples[train_samples['Label'] == 'N'].sample(10000),
                        train_samples[train_samples['Label'] == 'PVC'],
                        ]).sample(frac=1).reset_index(drop=True)
        # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(2000),
        #                 val_samples[val_samples['Label'] == 'PVC'],
        #                 ]).reset_index(drop=True)

        # train_df = pd.concat([train_samples[train_samples['Label'] == 'N'].sample(10000),
        #                       train_samples[train_samples['Label'] == 'PVC'],
        #                       ]).sample(frac=1).reset_index(drop=True)

        # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(500),
        #                     val_samples[val_samples['Label'] == 'PVC'],
        #                     ]).reset_index(drop=True)

        # df_name = '/datasets/extra_space2/ikachko/ecg/china_challenge/icbeb_peaks.csv'
        # df = pd.read_csv(df_name)

        df = pd.read_csv(df_name)
        # df = df[df['Dataset'] == 'physionet.org/files/svdb/1.0.0/']
        df = df[df['Dataset'] == 'physionet.org/files/mitdb/1.0.0/']
        df = df[df['Location'] > 201].reset_index(drop=True)
        # df = df[(df['Location']+201) < 230400].reset_index(drop=True) # mit-svdb
        df = df[(df['Location']+200) < 650000].reset_index(drop=True) #mit-bih

        val_df = pd.concat([df[df['Label'] == 'N'].sample(6000),
                            df[df['Label'] == 'PVC'],
                            ]).reset_index(drop=True)
        print(pd.value_counts(val_df['Label']))
        # val_df = pd.concat([df[df['Label'] == 'Normal'].sample(20000),
        #                     df[df['Label'] == 'PVC'],
        #                     ]).reset_index(drop=True)

        # val_df = pd.concat([val_samples[val_samples['Label'] == 'N'].sample(12000),
        #                     val_samples[val_samples['Label'] == 'PVC'],
        #                     ]).reset_index(drop=True)

        # val_df = pd.read_csv('/datasets/extra_space2/ikachko/ecg/PVC_mit_svdb.csv')
        # val_df = val_df[(val_df['Location']+200) < 230400].reset_index(drop=True) # mit-svdb
        # val_df = val_df[val_df['Location'] > 201].reset_index(drop=True)
        #
        # val_df = pd.concat([val_df[val_df['Label'] == 'N'].sample(12000),
        #                 val_df[val_df['Label'] == 'PVC'],
        #                 ]).reset_index(drop=True)

        # val_samples = df[df['PathToData'].isin(['A09.mat', 'A02.mat'])]
        # train_samples = df[~df['PathToData'].isin(['A09.mat', 'A02.mat'])]
        #
        # train_df = pd.concat([train_samples[train_samples['Label'] == 'Normal'].sample(30000),
        #                 train_samples[train_samples['Label'] == 'PVC'],
        #                 ]).sample(frac=1).reset_index(drop=True)
        # val_df = pd.concat([val_samples[val_samples['Label'] == 'Normal'].sample(5000),
        #                 val_samples[val_samples['Label'] == 'PVC'],
        #                 ]).reset_index(drop=True)


        # train_df, val_df = self.__train_val_split(df)
        # train_df.reset_index(inplace=True)
        # val_df.reset_index(inplace=True)
        print(pd.unique(train_df['PathToData']))
        print(pd.unique(val_df['PathToData']))

        # self.train_dataset = ECGPeaksDataset(exp_path, train_df, augmentation=False)
        # self.eval_train_dataset = ECGPeaksDataset(exp_path, train_df)
        # self.val_dataset = ECGPeaksDataset('/datasets/extra_space2/ikachko/ecg/china_challenge', val_df)
        self.train_dataset = ECGPeaksDatasetMIT(exp_path, train_df, augmentation=False)
        self.eval_train_dataset = ECGPeaksDatasetMIT(exp_path, train_df)
        self.val_dataset = ECGPeaksDatasetMIT(exp_path, val_df)
        # self.train_dataset = PQRSTDataset(exp_path, train_df, augmentation=False)
        # self.eval_train_dataset = PQRSTDataset(exp_path, train_df)
        # self.val_dataset = PQRSTDataset(exp_path, val_df)

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

    def Fbeta(self, r, p, beta):
        return torch.mean((1 + beta ** 2) * p * r / (beta ** 2 * p + r + 1e-20)).item()

    def __train_val_split(self, df):
        train_data, test_data = train_test_split(df, test_size=0.2,
                                                 random_state=42, stratify=df['Label'])
        return train_data, test_data


class InceptionTime(BaseLine):

    def __init__(self):
        super(InceptionTime, self).__init__()

        self.model = InceptionModel(
            # num_blocks=5,
            num_blocks=3,
            in_channels=1,
            out_channels=100,
            bottleneck_channels=1,
            kernel_sizes=[10, 20, 200],
            use_residuals=True,
            num_pred_classes=2
        )


class ResNet(BaseLine):

    def __init__(self):
        super(ResNet, self).__init__()

        self.model = ResNetBaseline(
            in_channels=1,
            num_pred_classes=2
        )


class RNNModel(BaseLine):
    def __init__(self):
        super(RNNModel, self).__init__()

        self.model = BidirectionalLSTM(
            input_size=240,
            output_size=2,
            hidden_dim=100,
            n_layers=3,
            drop_prob=0.5
        )


if __name__ == "__main__":

    from ignite.metrics import Accuracy, Recall, Fbeta, MetricsLambda

    model = RNNModel()

    model.prepare_data()

    tr = model.train_dataset.df
    val = model.val_dataset.df

    # print(pd.value_counts(tr['Label']))
    # print(pd.value_counts(val['Label']))

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
