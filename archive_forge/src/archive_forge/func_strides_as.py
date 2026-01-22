import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def strides_as(self, obj):
    """
        Return the strides tuple as an array of some other
        c-types type. For example: ``self.strides_as(ctypes.c_longlong)``.
        """
    if self._zerod:
        return None
    return (obj * self._arr.ndim)(*self._arr.strides)