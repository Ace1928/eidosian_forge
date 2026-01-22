import itertools
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import backend
from tensorflow.python.ops import array_ops
def normalize_padding(value):
    if isinstance(value, (list, tuple)):
        return value
    padding = value.lower()
    if padding not in {'valid', 'same', 'causal'}:
        raise ValueError('The `padding` argument must be a list/tuple or one of "valid", "same" (or "causal", only for `Conv1D). Received: ' + str(padding))
    return padding