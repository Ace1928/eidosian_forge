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
def rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars, forward_inputs):
    """Handles special case of IndexedSlices returned from while gradient.

  Some gradient functions return IndexedSlices instead of a Tensor (e.g. the
  gradient of Gather ops). When this happens in the gradient of a while body,
  the resulting gradient body function will have mismatched inputs and outputs,
  since the input is a single Tensor, but the IndexedSlices gets unnested into
  three output Tensors.

  This function fixes this by rewriting the gradient body to have three inputs
  to match the three outputs, i.e., it effectively converts the input Tensor
  into an input IndexedSlices. It also returns new `loop_vars` to reflect the
  new inputs.

  Args:
    grads: the input gradient Tensors to the while gradient computation.
    body_grad_graph: _WhileBodyGradFuncGraph.
    loop_vars: list of Tensors. The inputs to body_grad_graph.
    forward_inputs: list of Tensors. The (flat) inputs to the forward-pass While
      op.

  Returns:
    The new loop_vars to pass to body_grad_graph.
  """
    inputs_with_grads = [t for g, t in zip(grads, forward_inputs) if g is not None]
    structured_outputs = body_grad_graph.structured_outputs[3:]
    for forward_input, output in zip(inputs_with_grads, structured_outputs):
        if not isinstance(output, indexed_slices.IndexedSlices):
            continue
        if forward_input.dtype == dtypes.resource:
            loop_vars = _rewrite_input_as_indexed_slices(body_grad_graph, output, forward_input, loop_vars)
        else:
            _rewrite_output_as_tensor(body_grad_graph, output)
    return loop_vars