import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def byte_swap_tflite_model_obj(model, from_endiness, to_endiness):
    """Byte swaps the buffers field in a TFLite model.

  Args:
    model: TFLite model object of from_endiness format.
    from_endiness: The original endianness format of the buffers in model.
    to_endiness: The destined endianness format of the buffers in model.
  """
    if model is None:
        return
    buffer_swapped = []
    types_of_16_bits = [schema_fb.TensorType.FLOAT16, schema_fb.TensorType.INT16, schema_fb.TensorType.UINT16]
    types_of_32_bits = [schema_fb.TensorType.FLOAT32, schema_fb.TensorType.INT32, schema_fb.TensorType.COMPLEX64, schema_fb.TensorType.UINT32]
    types_of_64_bits = [schema_fb.TensorType.INT64, schema_fb.TensorType.FLOAT64, schema_fb.TensorType.COMPLEX128, schema_fb.TensorType.UINT64]
    for subgraph in model.subgraphs:
        for tensor in subgraph.tensors:
            if tensor.buffer > 0 and tensor.buffer < len(model.buffers) and (tensor.buffer not in buffer_swapped) and (model.buffers[tensor.buffer].data is not None):
                if tensor.type in types_of_16_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 2, from_endiness, to_endiness)
                elif tensor.type in types_of_32_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 4, from_endiness, to_endiness)
                elif tensor.type in types_of_64_bits:
                    byte_swap_buffer_content(model.buffers[tensor.buffer], 8, from_endiness, to_endiness)
                else:
                    continue
                buffer_swapped.append(tensor.buffer)