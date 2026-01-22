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
def train_tensorflow_mnist(num_workers: int=2, use_gpu: bool=False, epochs: int=4) -> Result:
    train_dataset = get_dataset(split_type='train')
    config = {'lr': 0.001, 'batch_size': 64, 'epochs': epochs}
    scaling_config = dict(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TensorflowTrainer(train_loop_per_worker=train_func, train_loop_config=config, datasets={'train': train_dataset}, scaling_config=scaling_config)
    results = trainer.fit()
    print(results.metrics)
    return results