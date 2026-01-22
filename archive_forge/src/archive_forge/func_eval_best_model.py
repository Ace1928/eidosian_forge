import argparse
import os
import numpy as np
import torch
import torch.optim as optim
from ray.tune.examples.mnist_pytorch import test_func, ConvNet, get_data_loaders
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
def eval_best_model(results: tune.ResultGrid):
    """Test the best model given output of tuner.fit()."""
    with results.get_best_result().checkpoint.as_directory() as best_checkpoint_path:
        best_model = ConvNet()
        best_checkpoint = torch.load(os.path.join(best_checkpoint_path, 'checkpoint.pt'))
        best_model.load_state_dict(best_checkpoint['model'])
        test_acc = test_func(best_model, get_data_loaders()[1])
        print('best model accuracy: ', test_acc)