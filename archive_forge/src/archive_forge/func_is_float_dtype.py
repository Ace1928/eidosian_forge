import numpy as np
from keras.src.api_export import keras_export
from keras.src.backend import config
from keras.src.backend.common import global_state
from keras.src.backend.common.name_scope import current_path
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.utils.module_utils import tensorflow as tf
from keras.src.utils.naming import auto_name
@keras_export('keras.backend.is_float_dtype')
def is_float_dtype(dtype):
    dtype = standardize_dtype(dtype)
    return dtype.startswith('float') or dtype.startswith('bfloat')