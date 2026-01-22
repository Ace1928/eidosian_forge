import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def byte_swap_tflite_buffer(tflite_model, from_endiness, to_endiness):
    """Generates a new model byte array after byte swapping its buffers field.

  Args:
    tflite_model: TFLite flatbuffer in a byte array.
    from_endiness: The original endianness format of the buffers in
      tflite_model.
    to_endiness: The destined endianness format of the buffers in tflite_model.

  Returns:
    TFLite flatbuffer in a byte array, after being byte swapped to to_endiness
    format.
  """
    if tflite_model is None:
        return None
    model = convert_bytearray_to_object(tflite_model)
    byte_swap_tflite_model_obj(model, from_endiness, to_endiness)
    return convert_object_to_bytearray(model)