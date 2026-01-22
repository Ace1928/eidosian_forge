import argparse
import time
from ray import train, tune
from ray.tune.logger import LoggerCallback
def trial_str_creator(trial):
    return '{}_{}_123'.format(trial.trainable_name, trial.trial_id)