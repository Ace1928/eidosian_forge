from __future__ import annotations
import sys
import math
def is_array_api_obj(x):
    """
    Check if x is an array API compatible array object.
    """
    return _is_numpy_array(x) or _is_cupy_array(x) or _is_torch_array(x) or hasattr(x, '__array_namespace__')