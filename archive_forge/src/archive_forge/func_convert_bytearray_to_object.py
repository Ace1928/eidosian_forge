import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def convert_bytearray_to_object(model_bytearray):
    """Converts a tflite model from a bytearray to an object for parsing."""
    model_object = schema_fb.Model.GetRootAsModel(model_bytearray, 0)
    return schema_fb.ModelT.InitFromObj(model_object)