import io
import pathlib
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from tensorflow.python.util.tf_export import keras_export
def get_interpolation(interpolation):
    interpolation = interpolation.lower()
    if interpolation not in _TF_INTERPOLATION_METHODS:
        raise NotImplementedError('Value not recognized for `interpolation`: {}. Supported values are: {}'.format(interpolation, _TF_INTERPOLATION_METHODS.keys()))
    return _TF_INTERPOLATION_METHODS[interpolation]