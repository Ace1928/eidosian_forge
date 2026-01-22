import argparse
import tensorflow as tf
import ray
from ray import train
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.data.preprocessors import Concatenator
from ray.train import Result, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer
def train_tensorflow_regression(num_workers: int=2, use_gpu: bool=False) -> Result:
    dataset = ray.data.read_csv('s3://anonymous@air-example-data/regression.csv')
    preprocessor = Concatenator(exclude=['', 'y'], output_column_name='x')
    dataset = preprocessor.fit_transform(dataset)
    config = {'lr': 0.001, 'batch_size': 32, 'epochs': 4}
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TensorflowTrainer(train_loop_per_worker=train_func, train_loop_config=config, scaling_config=scaling_config, datasets={'train': dataset})
    results = trainer.fit()
    print(results.metrics)
    return results