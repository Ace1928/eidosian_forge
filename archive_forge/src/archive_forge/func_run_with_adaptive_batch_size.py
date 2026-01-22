import re
import warnings
import keras_tuner
import tensorflow as tf
from packaging.version import parse
from tensorflow import nest
def run_with_adaptive_batch_size(batch_size, func, **fit_kwargs):
    x = fit_kwargs.pop('x')
    validation_data = None
    if 'validation_data' in fit_kwargs:
        validation_data = fit_kwargs.pop('validation_data')
    while batch_size > 0:
        try:
            history = func(x=x, validation_data=validation_data, **fit_kwargs)
            break
        except tf.errors.ResourceExhaustedError as e:
            if batch_size == 1:
                raise e
            batch_size //= 2
            print('Not enough memory, reduce batch size to {batch_size}.'.format(batch_size=batch_size))
            x = x.unbatch().batch(batch_size)
            if validation_data is not None:
                validation_data = validation_data.unbatch().batch(batch_size)
    return history