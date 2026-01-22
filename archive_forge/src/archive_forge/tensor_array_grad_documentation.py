from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
Gradient for TensorArraySplit.

  Args:
    op: Forward TensorArraySplit op.
    flow: Gradient `Tensor` flow to TensorArraySplit.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  