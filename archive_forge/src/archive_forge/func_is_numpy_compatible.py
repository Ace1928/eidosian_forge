import numpy as np
from . import pywrap_tensorflow
from tensorboard.compat.proto import types_pb2
@property
def is_numpy_compatible(self):
    return self._type_enum not in _NUMPY_INCOMPATIBLE