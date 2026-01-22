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
def score_trial(self, trial):
    """Score a completed `Trial`.

        This method can be overridden in subclasses to provide a score for
        a set of hyperparameter values. This method is called from `end_trial`
        on completed `Trial`s.

        Args:
            trial: A completed `Trial` object.
        """
    trial.score = trial.metrics.get_best_value(self.objective.name)
    trial.best_step = trial.metrics.get_best_step(self.objective.name)