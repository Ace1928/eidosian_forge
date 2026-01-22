import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def convert_object_to_bytearray(model_object, extra_buffer=b''):
    """Converts a tflite model from an object to a immutable bytearray."""
    builder = flatbuffers.Builder(1024)
    model_offset = model_object.Pack(builder)
    builder.Finish(model_offset, file_identifier=_TFLITE_FILE_IDENTIFIER)
    model_bytearray = bytes(builder.Output())
    model_bytearray = model_bytearray + extra_buffer
    return model_bytearray