import contextlib
from tensorflow.python import pywrap_tfe
def record_operation_forwardprop_only(op_type, output_tensors, input_tensors, backward_function, forwardprop_output_indices):
    """Records the operation on all forward accumulators in the stack.

  Args:
    op_type: a string for the operation type, used in the backprop code
    output_tensors: a list of Python Tensor objects output by the operation
    input_tensors: a list of input Tensors to the recorded operation
    backward_function: the function to be called to, given the gradients of the
      output tensors, produce the gradients of the input tensors. This function
      is automatically transposed to produce output gradients given input
      gradients.
    forwardprop_output_indices: indicates any output_tensors which contain JVPs.
      Typically these will have come from TFE_Py_PackForwardGradients. May be
      None or an empty sequence if there are no JVP outputs from the operation.
  """
    pywrap_tfe.TFE_Py_TapeSetRecordOperationForwardprop(op_type, output_tensors, input_tensors, backward_function, forwardprop_output_indices)