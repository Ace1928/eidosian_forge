import os
import tempfile
import pytorch_lightning as pl
import mlflow
from ray import train, tune
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.examples.mnist_ptl_mini import LightningMNISTClassifier, MNISTDataModule
An example showing how to use Pytorch Lightning training, Ray Tune
HPO, and MLflow autologging all together.