from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat
def make_int(v, arg_name):
    if isinstance(v, str):
        raise TypeError("Expected int for argument '%s' not %s." % (arg_name, repr(v)))
    try:
        return int(v)
    except (ValueError, TypeError):
        raise TypeError("Expected int for argument '%s' not %s." % (arg_name, repr(v)))