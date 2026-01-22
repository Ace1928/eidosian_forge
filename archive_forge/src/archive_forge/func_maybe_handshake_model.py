import contextlib
import copy
from io import BytesIO
from typing import Any, Dict, List, Optional
import catalogue
import numpy
from ..backends import Ops, get_current_ops
from ..compat import cupy, h5py
from ..compat import tensorflow as tf
from ..optimizers import Optimizer
from ..types import ArgsKwargs, ArrayXd
from ..util import get_array_module
from .shim import Shim
def maybe_handshake_model(keras_model):
    """Call the required predict/compile/build APIs to initialize a model if it
    is a subclass of tf.keras.Model. This is required to be able to call set_weights
    on subclassed layers."""
    try:
        keras_model.get_config()
        return keras_model
    except (AttributeError, NotImplementedError):
        pass
    for prop_name in ['catalogue_name', 'eg_x', 'eg_y', 'eg_shape']:
        if not hasattr(keras_model, prop_name):
            raise ValueError("Keras subclassed models are not whole-model serializable by TensorFlow. To work around this, you must decorate your keras model subclasses with the 'keras_subclass' decorator. The decorator requires a single X/Y input of fake-data that can be used to initialize your subclass model properly when loading the saved version.")
    ops: Ops = get_current_ops()
    if ops.device_type == 'cpu':
        device = 'CPU'
    else:
        device = tf.test.gpu_device_name()
    compile_args = keras_model.eg_compile
    with tf.device(device):
        keras_model.compile(**compile_args)
        keras_model.build(keras_model.eg_shape)
        keras_model.predict(keras_model.eg_x)
        if hasattr(keras_model, '_make_train_function'):
            keras_model._make_train_function()
        else:
            keras_model.make_train_function()
    return keras_model