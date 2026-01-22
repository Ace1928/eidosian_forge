import functools
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export
def tiled_loop_body(j):
    offset = j * parallel_iterations + num_remaining_iterations

    def tiled_loop_fn(i, pfor_config=None):
        if loop_fn_has_config:
            loop_fn_outputs = loop_fn(i + offset, pfor_config=pfor_config)
        else:
            loop_fn_outputs = loop_fn(i + offset)
        return nest.flatten(nest.map_structure(_composite_to_tensors, loop_fn_outputs))
    return _pfor_impl(tiled_loop_fn, parallel_iterations, fallback_to_while_loop=fallback_to_while_loop, pfor_config=pfor_config)