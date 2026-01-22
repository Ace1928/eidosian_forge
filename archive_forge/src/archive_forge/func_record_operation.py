import contextlib
from tensorflow.python import pywrap_tfe
def record_operation(op_type, output_tensors, input_tensors, backward_function, forward_function=None):
    """Records the operation on all tapes in the stack."""
    pywrap_tfe.TFE_Py_TapeSetRecordOperation(op_type, output_tensors, input_tensors, backward_function, forward_function)