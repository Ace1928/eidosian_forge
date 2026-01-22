import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def fit_with_adaptive_batch_size(model, batch_size, **fit_kwargs):
    history = run_with_adaptive_batch_size(batch_size, lambda **kwargs: model.fit(**kwargs), **fit_kwargs)
    return (model, history)