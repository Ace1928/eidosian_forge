import numpy as np
from . import pywrap_tensorflow
from tensorboard.compat.proto import types_pb2
@property
def is_quantized(self):
    """Returns whether this is a quantized data type."""
    return self.base_dtype in _QUANTIZED_DTYPES_NO_REF