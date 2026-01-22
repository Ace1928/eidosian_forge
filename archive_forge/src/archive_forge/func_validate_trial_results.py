import collections
import statistics
import numpy as np
from keras_tuner.src import backend
from keras_tuner.src import errors
from keras_tuner.src.backend import config
from keras_tuner.src.backend import keras
from keras_tuner.src.engine import hyperparameters as hp_module
from keras_tuner.src.engine import objective as obj_module
def validate_trial_results(results, objective, func_name):
    if isinstance(results, list):
        for elem in results:
            validate_trial_results(elem, objective, func_name)
        return
    if isinstance(results, (int, float, np.floating)):
        return
    if results is None:
        raise errors.FatalTypeError(f'The return value of {func_name} is None. Did you forget to return the metrics? ')
    if isinstance(objective, obj_module.DefaultObjective) and (not (isinstance(results, dict) and objective.name in results)):
        raise errors.FatalTypeError(f'Expected the return value of {func_name} to be a single float when `objective` is left unspecified. Recevied return value: {results} of type {type(results)}.')
    if isinstance(results, dict):
        if objective.name not in results:
            raise errors.FatalValueError(f'Expected the returned dictionary from {func_name} to have the specified objective, {objective.name}, as one of the keys. Received: {results}.')
        return
    if isinstance(results, keras.callbacks.History):
        return
    raise errors.FatalTypeError(f'Expected the return value of {func_name} to be one of float, dict, keras.callbacks.History, or a list of one of these types. Recevied return value: {results} of type {type(results)}.')