import argparse
import json
import os
import random
import tempfile
import numpy as np
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import PopulationBasedTraining
def pbt_function(config):
    """Toy PBT problem for benchmarking adaptive learning rate.

    The goal is to optimize this trainable's accuracy. The accuracy increases
    fastest at the optimal lr, which is a function of the current accuracy.

    The optimal lr schedule for this problem is the triangle wave as follows.
    Note that many lr schedules for real models also follow this shape:

     best lr
      ^
      |    /      |   /        |  /          | /            ------------> accuracy

    In this problem, using PBT with a population of 2-4 is sufficient to
    roughly approximate this lr schedule. Higher population sizes will yield
    faster convergence. Training will not converge without PBT.
    """
    lr = config['lr']
    checkpoint_interval = config.get('checkpoint_interval', 1)
    accuracy = 0.0
    step = 1
    checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            with open(os.path.join(checkpoint_dir, 'checkpoint.json'), 'r') as f:
                checkpoint_dict = json.load(f)
        accuracy = checkpoint_dict['acc']
        last_step = checkpoint_dict['step']
        step = last_step + 1
    midpoint = 100
    q_tolerance = 3
    noise_level = 2
    while True:
        if accuracy < midpoint:
            optimal_lr = 0.01 * accuracy / midpoint
        else:
            optimal_lr = 0.01 - 0.01 * (accuracy - midpoint) / midpoint
        optimal_lr = min(0.01, max(0.001, optimal_lr))
        q_err = max(lr, optimal_lr) / min(lr, optimal_lr)
        if q_err < q_tolerance:
            accuracy += 1.0 / q_err * random.random()
        elif lr > optimal_lr:
            accuracy -= (q_err - q_tolerance) * random.random()
        accuracy += noise_level * np.random.normal()
        accuracy = max(0, accuracy)
        metrics = {'mean_accuracy': accuracy, 'cur_lr': lr, 'optimal_lr': optimal_lr, 'q_err': q_err, 'done': accuracy > midpoint * 2}
        if step % checkpoint_interval == 0:
            with tempfile.TemporaryDirectory() as tempdir:
                with open(os.path.join(tempdir, 'checkpoint.json'), 'w') as f:
                    checkpoint_dict = {'acc': accuracy, 'step': step}
                    json.dump(checkpoint_dict, f)
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)
        step += 1