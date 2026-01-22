import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_string_ops import *
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@tf_export(v1=['strings.to_hash_bucket', 'string_to_hash_bucket'])
@dispatch.register_unary_elementwise_api
@dispatch.add_dispatch_support
def string_to_hash_bucket_v1(string_tensor=None, num_buckets=None, name=None, input=None):
    string_tensor = deprecation.deprecated_argument_lookup('input', input, 'string_tensor', string_tensor)
    return gen_string_ops.string_to_hash_bucket(string_tensor, num_buckets, name)