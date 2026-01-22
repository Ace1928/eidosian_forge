import functools
import threading
import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.utils import control_flow_util
from keras.src.utils import tf_inspect
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
def infer_init_val_and_dtype(initializer, dtype, shape, layout=None):
    if initializer is not None and (not callable(initializer)):
        init_val = initializer
        variable_dtype = None
    else:
        if tf_inspect.isclass(initializer):
            initializer = initializer()
        if layout:
            init_val = functools.partial(initializer, shape, dtype=dtype, layout=layout)
        else:
            init_val = functools.partial(initializer, shape, dtype=dtype)
        variable_dtype = dtype.base_dtype
    return (init_val, variable_dtype)