import collections
from functools import partial
import string
import sys
import traceback
import numpy as np
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
@def_function.function
def while_fn():
    init_values = self._init_values()
    ta_shape_invariants = [tensor_shape.TensorShape([]) for _ in self._pfor_input.outputs]
    shape_invariants = [tensor_shape.TensorShape([]), tensor_shape.TensorShape([None])] + output_shapes + ta_shape_invariants
    while_outputs = while_loop.while_loop(cond, body, init_values, shape_invariants=shape_invariants, parallel_iterations=self._parallel_iterations)
    if indices_to_stack:
        return while_outputs
    else:
        num_inputs = self._pfor_input.num_inputs
        new_inputs = while_outputs[2:num_inputs + 2]
        output_tas = while_outputs[num_inputs + 2:]
        assert cond_is_stacked[0] is not None
        outputs = []
        for i, inp in enumerate(new_inputs):
            if cond_is_stacked[0]:
                if i in self._body_pass_through_indices:
                    outputs.append(init_values[i + 2])
                else:
                    ta = output_tas[i]
                    if _variant_type_id(inp) == full_type_pb2.TFT_ARRAY:
                        shape_and_type = _parse_variant_shapes_and_types(inp)[0]
                        length = list_ops.tensor_list_length(inp)

                        def _stack_loop_body(index, output_list):
                            current_value = ta.read(index)
                            output_list = list_ops.tensor_list_set_item(output_list, index, list_ops.tensor_list_stack(current_value, shape_and_type.dtype))
                            return (index + 1, output_list)
                        output_list = list_ops.tensor_list_reserve(tensor_shape.TensorShape(shape_and_type.shape), length, shape_and_type.dtype)
                        _, output_list = while_loop.while_loop(lambda index, _: index < length, _stack_loop_body, [0, output_list])
                        outputs.append(output_list)
                    else:
                        outputs.append(ta.stack())
            else:
                outputs.append(inp)
        return outputs