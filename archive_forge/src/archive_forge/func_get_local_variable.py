import copy
import enum
import functools
import sys
import threading
import traceback
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import function_utils
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['get_local_variable'])
def get_local_variable(name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=False, collections=None, caching_device=None, partitioner=None, validate_shape=True, use_resource=None, custom_getter=None, constraint=None, synchronization=VariableSynchronization.AUTO, aggregation=VariableAggregation.NONE):
    if collections:
        collections += [ops.GraphKeys.LOCAL_VARIABLES]
    else:
        collections = [ops.GraphKeys.LOCAL_VARIABLES]
    return get_variable(name, shape=shape, dtype=dtype, initializer=initializer, regularizer=regularizer, trainable=False, collections=collections, caching_device=caching_device, partitioner=partitioner, validate_shape=validate_shape, use_resource=use_resource, synchronization=synchronization, aggregation=aggregation, custom_getter=custom_getter, constraint=constraint)