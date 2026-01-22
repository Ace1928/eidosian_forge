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
def tune_mnist(num_samples=10, num_epochs=10, gpus_per_trial=0):
    config = {'layer_1': tune.choice([32, 64, 128]), 'layer_2': tune.choice([64, 128, 256]), 'lr': tune.loguniform(0.0001, 0.1), 'batch_size': tune.choice([32, 64, 128])}
    trainable = tune.with_parameters(train_mnist_tune, num_epochs=num_epochs, num_gpus=gpus_per_trial)
    tuner = tune.Tuner(tune.with_resources(trainable, resources={'cpu': 1, 'gpu': gpus_per_trial}), tune_config=tune.TuneConfig(metric='loss', mode='min', num_samples=num_samples), run_config=train.RunConfig(name='tune_mnist'), param_space=config)
    results = tuner.fit()
    print('Best hyperparameters found were: ', results.get_best_result().config)