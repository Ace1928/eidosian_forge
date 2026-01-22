import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def get_small_mlp(num_hidden, num_classes, input_dim):
    """Get a small mlp of the model type specified by `get_model_type`."""
    model_type = get_model_type()
    if model_type == 'subclass':
        return get_small_subclass_mlp(num_hidden, num_classes)
    if model_type == 'subclass_custom_build':
        return get_small_subclass_mlp_with_custom_build(num_hidden, num_classes)
    if model_type == 'sequential':
        return get_small_sequential_mlp(num_hidden, num_classes, input_dim)
    if model_type == 'functional':
        return get_small_functional_mlp(num_hidden, num_classes, input_dim)
    raise ValueError('Unknown model type {}'.format(model_type))