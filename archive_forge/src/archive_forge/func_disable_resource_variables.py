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
@deprecation.deprecated(None, 'non-resource variables are not supported in the long term')
@tf_export(v1=['disable_resource_variables'])
def disable_resource_variables():
    """Opts out of resource variables.

  If your code needs tf.disable_resource_variables() to be called to work
  properly please file a bug.
  """
    global _DEFAULT_USE_RESOURCE
    _DEFAULT_USE_RESOURCE = False
    logging.vlog(1, 'Disabling resource variables')
    _api_usage_gauge.get_cell().set(False)