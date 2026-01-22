import collections
import hashlib
import os
import random
import threading
import warnings
from datetime import datetime
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
from keras_tuner.src.engine import stateful
from keras_tuner.src.engine import trial as trial_module
def show_hyperparameter_table(self, trial):
    template = '{{0:{0}}}|{{1:{0}}}|{{2}}'.format(self.col_width)
    best_trials = self.oracle.get_best_trials()
    best_trial = best_trials[0] if len(best_trials) > 0 else None
    if trial.hyperparameters.values:
        print(template.format('Value', 'Best Value So Far', 'Hyperparameter'))
        for hp, value in trial.hyperparameters.values.items():
            best_value = best_trial.hyperparameters.values.get(hp) if best_trial else '?'
            print(template.format(self.format_value(value), self.format_value(best_value), hp))
    else:
        print('default configuration')