import math
import os
import torch
from filelock import FileLock
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import MNIST
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray import train, tune
def train_mnist_tune(config, num_epochs=10, num_gpus=0):
    data_dir = os.path.abspath('./data')
    model = LightningMNISTClassifier(config, data_dir)
    with FileLock(os.path.expanduser('~/.data.lock')):
        dm = MNISTDataModule(data_dir=data_dir, batch_size=config['batch_size'])
    metrics = {'loss': 'ptl/val_loss', 'acc': 'ptl/val_accuracy'}
    trainer = pl.Trainer(max_epochs=num_epochs, gpus=math.ceil(num_gpus), enable_progress_bar=False, callbacks=[TuneReportCheckpointCallback(metrics, on='validation_end', save_checkpoints=False)])
    trainer.fit(model, dm)