import copy
import math
from keras_tuner.src import backend
from keras_tuner.src import utils
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.engine import oracle as oracle_module
from keras_tuner.src.engine import tuner as tuner_module
def run_trial(self, trial, *fit_args, **fit_kwargs):
    hp = trial.hyperparameters
    if 'tuner/epochs' in hp.values:
        fit_kwargs['epochs'] = hp.values['tuner/epochs']
        fit_kwargs['initial_epoch'] = hp.values['tuner/initial_epoch']
    return super().run_trial(trial, *fit_args, **fit_kwargs)