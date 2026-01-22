import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def evaluate_with_adaptive_batch_size(model, batch_size, verbose=1, **fit_kwargs):
    return run_with_adaptive_batch_size(batch_size, lambda x, validation_data, **kwargs: model.evaluate(x, verbose=verbose, **kwargs), **fit_kwargs)