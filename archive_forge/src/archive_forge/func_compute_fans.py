import math
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.backend import random
from keras.src.initializers.initializer import Initializer
from keras.src.saving import serialization_lib
def compute_fans(shape):
    """Computes the number of input and output units for a weight shape.

    Args:
        shape: Integer shape tuple.

    Returns:
        A tuple of integer scalars: `(fan_in, fan_out)`.
    """
    shape = tuple(shape)
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return (int(fan_in), int(fan_out))