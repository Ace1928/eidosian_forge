from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest
def rewrite(old_output, new_input):
    assert old_output.type == 'Identity'
    concat_op = old_output.inputs[0].op
    assert concat_op.type == 'ConcatV2'
    old_concat_args = concat_op.inputs[:-1]
    return array_ops.concat([new_input] + old_concat_args[1:], 0)