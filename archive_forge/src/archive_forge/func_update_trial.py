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
@synchronized
def update_trial(self, trial_id, metrics, step=0):
    """Used by a worker to report the status of a trial.

        Args:
            trial_id: A string, a previously seen trial id.
            metrics: Dict. The keys are metric names, and the values are this
                trial's metric values.
            step: Optional float, reporting intermediate results. The current
                value in a timeseries representing the state of the trial. This
                is the value that `metrics` will be associated with.

        Returns:
            Trial object.
        """
    trial = self.trials[trial_id]
    self._check_objective_found(metrics)
    for metric_name, metric_value in metrics.items():
        if not trial.metrics.exists(metric_name):
            direction = _maybe_infer_direction_from_objective(self.objective, metric_name)
            trial.metrics.register(metric_name, direction=direction)
        trial.metrics.update(metric_name, metric_value, step=step)
    self._save_trial(trial)
    return trial