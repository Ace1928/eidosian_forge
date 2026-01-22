import collections
import numpy as np
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export('debugging.assert_none_equal', v1=[])
@dispatch.register_binary_elementwise_assert_api
@dispatch.add_dispatch_support
@_binary_assert_doc_v2('!=', 'assert_none_equal', 6)
def assert_none_equal_v2(x, y, summarize=None, message=None, name=None):
    return assert_none_equal(x=x, y=y, summarize=summarize, message=message, name=name)