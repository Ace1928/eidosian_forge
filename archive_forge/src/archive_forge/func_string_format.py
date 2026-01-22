import typing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_array_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import compat as util_compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
@dispatch.dispatch_for_api(string_ops.string_format)
def string_format(template: str, inputs: typing.Union[ragged_tensor.Ragged, typing.List[ragged_tensor.RaggedOrDense]], placeholder='{}', summarize=3, name=None):
    """Version of tf.strings.format that handles RaggedTensors."""
    if tensor_util.is_tf_type(inputs) or ragged_tensor.is_ragged(inputs):
        inputs = [inputs]
    split_template = template.split(placeholder)
    if len(inputs) != len(split_template) - 1:
        raise ValueError('num placeholders in template and num inputs must match: {} vs {}'.format(len(split_template) - 1, len(inputs)))
    with ops.name_scope(name, 'StringFormat', [inputs]):
        output_pieces = [constant_op.constant(split_template[0])]
        for i, input in enumerate(inputs):
            if ragged_tensor.is_ragged(input):
                output_pieces.append(ragged_tensor_to_string(input, summarize))
            else:
                output_pieces.append(string_ops.string_format('{}', [input], summarize=summarize))
            output_pieces.append(constant_op.constant(split_template[i + 1]))
        if len(output_pieces) == 1:
            return output_pieces[0]
        else:
            return string_ops.reduce_join(output_pieces)