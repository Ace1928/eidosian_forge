import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
def average_metrics_dicts(metrics_dicts):
    """Averages the metrics dictionaries to one metrics dictionary."""
    metrics = collections.defaultdict(list)
    for metrics_dict in metrics_dicts:
        for metric_name, metric_value in metrics_dict.items():
            metrics[metric_name].append(metric_value)
    averaged_metrics = {metric_name: np.mean(metric_values) for metric_name, metric_values in metrics.items()}
    return averaged_metrics