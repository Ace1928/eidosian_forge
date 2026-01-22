import collections
import copy
import numpy as np
from tensorflow.python.data.experimental.ops import cardinality
from tensorflow.python.distribute.coordinator import cluster_coordinator as coordinator_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_value
from tensorflow.python.util import nest
def assert_no_legacy_layers(layers):
    """Prevent tf.layers.Layers from being used with Keras.

  Certain legacy layers inherit from their keras analogs; however they are
  not supported with keras and can lead to subtle and hard to diagnose bugs.

  Args:
    layers: A list of layers to check

  Raises:
    TypeError: If any elements of layers are tf.layers.Layers
  """
    legacy_layers = [l for l in layers if getattr(l, '_is_legacy_layer', None)]
    if legacy_layers:
        layer_str = '\n'.join(('  ' + str(l) for l in legacy_layers))
        raise TypeError('The following are legacy tf.layers.Layers:\n{}\nTo use keras as a framework (for instance using the Network, Model, or Sequential classes), please use the tf.keras.layers implementation instead. (Or, if writing custom layers, subclass from tf.keras.layers rather than tf.layers)'.format(layer_str))