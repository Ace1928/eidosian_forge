import re
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
@tf_export('map_fn', v1=[])
@deprecation.deprecated_arg_values(None, 'back_prop=False is deprecated. Consider using tf.stop_gradient instead.\nInstead of:\nresults = tf.map_fn(fn, elems, back_prop=False)\nUse:\nresults = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))', warn_once=True, back_prop=False)
@deprecation.deprecated_args(None, 'Use fn_output_signature instead', 'dtype')
def map_fn_v2(fn, elems, dtype=None, parallel_iterations=None, back_prop=True, swap_memory=False, infer_shape=True, name=None, fn_output_signature=None):
    """Transform `elems` by applying `fn` to each element unstacked on axis 0."""
    if fn_output_signature is None:
        fn_output_signature = dtype
    return map_fn(fn=fn, elems=elems, fn_output_signature=fn_output_signature, parallel_iterations=parallel_iterations, back_prop=back_prop, swap_memory=swap_memory, infer_shape=infer_shape, name=name)