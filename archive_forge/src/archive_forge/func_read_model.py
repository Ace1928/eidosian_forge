import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def read_model(input_tflite_file):
    """Reads a tflite model as a python object.

  Args:
    input_tflite_file: Full path name to the input tflite file

  Raises:
    RuntimeError: If input_tflite_file path is invalid.
    IOError: If input_tflite_file cannot be opened.

  Returns:
    A python object corresponding to the input tflite file.
  """
    if not gfile.Exists(input_tflite_file):
        raise RuntimeError('Input file not found at %r\n' % input_tflite_file)
    with gfile.GFile(input_tflite_file, 'rb') as input_file_handle:
        model_bytearray = bytearray(input_file_handle.read())
    model = convert_bytearray_to_object(model_bytearray)
    if sys.byteorder == 'big':
        byte_swap_tflite_model_obj(model, 'little', 'big')
    return model