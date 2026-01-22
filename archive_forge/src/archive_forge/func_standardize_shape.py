import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
def standardize_shape(shape):
    if not isinstance(shape, tuple):
        if shape is None:
            raise ValueError('Undefined shapes are not supported.')
        if not hasattr(shape, '__iter__'):
            raise ValueError(f"Cannot convert '{shape}' to a shape.")
        if config.backend() == 'tensorflow':
            if isinstance(shape, tf.TensorShape):
                shape = shape.as_list()
        shape = tuple(shape)
    if config.backend() == 'torch':
        shape = tuple(map(lambda x: int(x) if x is not None else None, shape))
    for e in shape:
        if e is None:
            continue
        if config.backend() == 'jax' and str(e) == 'b':
            continue
        if not is_int_dtype(type(e)):
            raise ValueError(f"Cannot convert '{shape}' to a shape. Found invalid entry '{e}' of type '{type(e)}'. ")
        if e < 0:
            raise ValueError(f"Cannot convert '{shape}' to a shape. Negative dimensions are not allowed.")
    return shape