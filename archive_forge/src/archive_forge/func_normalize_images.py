import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import ray
from ray import train
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.data.datasource import SimpleTensorFlowDatasource
from ray.data.extensions import TensorArray
from ray.train import Result
from ray.train.tensorflow import TensorflowTrainer, prepare_dataset_shard
def normalize_images(x):
    x = np.float32(x.numpy()) / 255.0
    x = np.reshape(x, (-1,))
    return x