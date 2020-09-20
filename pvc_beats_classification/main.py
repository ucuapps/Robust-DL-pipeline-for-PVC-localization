import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from modules.segmentation_baseline import SegmentationBaseLine
from modules.baseline import BaseLine, InceptionTime, RNNModel, ResNet

def main():
    # model = SegmentationBaseLine()
    model = RNNModel()#ResNet()#InceptionTime() #RNNModel()
    logger = TensorBoardLogger(save_dir='logs', name='TIME_SVDB_BIH')

    trainer = pl.Trainer(gpus=[1], logger=logger, progress_bar_refresh_rate=1, checkpoint_callback=True)

    trainer.fit(model)


if __name__ == "__main__":
    main()
