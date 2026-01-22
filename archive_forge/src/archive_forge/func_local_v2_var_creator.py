import abc
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.dtensor import dtensor_api as dtensor
from keras.src.dtensor import utils as dtensor_utils
from keras.src.engine import base_layer
from keras.src.engine import base_layer_utils
from keras.src.engine import keras_tensor
from keras.src.saving.legacy.saved_model import metric_serialization
from keras.src.utils import generic_utils
from keras.src.utils import losses_utils
from keras.src.utils import metrics_utils
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
def local_v2_var_creator(initializer=None, dtype=None, shape=None, **kwargs):
    init_val, var_dtype = base_layer_utils.infer_init_val_and_dtype(initializer, dtype, shape)
    v1_only_args = ['use_resource', 'collections']
    for v1_arg in v1_only_args:
        kwargs.pop(v1_arg, None)
    kwargs['experimental_enable_variable_lifting'] = False
    return tf.Variable(initial_value=init_val, dtype=var_dtype, shape=shape, **kwargs)