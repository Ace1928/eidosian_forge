import os
import random
import argparse
import pandas as pd
from datetime import datetime
from ray.tune import run, sample_from
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.schedulers.pb2 import PB2
def explore(config):
    if config['train_batch_size'] < config['sgd_minibatch_size'] * 2:
        config['train_batch_size'] = config['sgd_minibatch_size'] * 2
    if config['lambda'] > 1:
        config['lambda'] = 1
    config['train_batch_size'] = int(config['train_batch_size'])
    return config