import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
def convert_to_metrics_dict(results, objective):
    """Convert any supported results type to a metrics dictionary."""
    if isinstance(results, list):
        return average_metrics_dicts([convert_to_metrics_dict(elem, objective) for elem in results])
    if isinstance(results, (int, float, np.floating)):
        return {objective.name: float(results)}
    if isinstance(results, dict):
        return results
    if isinstance(results, keras.callbacks.History):
        best_value, _ = _get_best_value_and_best_epoch_from_history(results, objective)
        return best_value