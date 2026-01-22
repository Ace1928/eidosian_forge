import copy
import random
import re
import struct
import sys
import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema_fb
from tensorflow.lite.python import schema_util
from tensorflow.python.platform import gfile
def type_to_name(tensor_type):
    """Converts a numerical enum to a readable tensor type."""
    for name, value in schema_fb.TensorType.__dict__.items():
        if value == tensor_type:
            return name
    return None