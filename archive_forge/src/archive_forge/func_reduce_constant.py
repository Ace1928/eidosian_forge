from typing import Any, Dict
import numpy as np
from onnx.onnx_pb import NodeProto
from onnx.reference.op_run import OpRun, RuntimeTypeError
def reduce_constant(self, data, const_val, axes, keepdims):
    """Special case reduction where the output value is a constant."""
    output_shape = self.output_shape(data, axes, keepdims)
    return (np.full(output_shape, const_val, dtype=data.dtype),)