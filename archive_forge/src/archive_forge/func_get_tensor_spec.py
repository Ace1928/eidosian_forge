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
def get_tensor_spec(t, dynamic_batch=False, name=None):
    """Returns a `TensorSpec` given a single `Tensor` or `TensorSpec`."""
    if isinstance(t, type_spec.TypeSpec):
        spec = t
    elif is_extension_type(t):
        spec = t._type_spec
    elif hasattr(t, '_keras_history') and hasattr(t._keras_history[0], '_type_spec'):
        return t._keras_history[0]._type_spec
    elif hasattr(t, 'shape') and hasattr(t, 'dtype'):
        spec = tensor_lib.TensorSpec(shape=t.shape, dtype=t.dtype, name=name)
    else:
        return None
    if not dynamic_batch:
        return spec
    dynamic_batch_spec = copy.deepcopy(spec)
    shape = dynamic_batch_spec._shape
    if shape.rank is not None and shape.rank > 0:
        shape_list = shape.as_list()
        shape_list[0] = None
        dynamic_batch_spec._shape = tensor_shape.TensorShape(shape_list)
    return dynamic_batch_spec