import collections
import functools
import os
from .._utils import set_module
from .._utils._inspect import getargspec
from numpy.core._multiarray_umath import (
def set_array_function_like_doc(public_api):
    if public_api.__doc__ is not None:
        public_api.__doc__ = public_api.__doc__.replace('${ARRAY_FUNCTION_LIKE}', array_function_like_doc)
    return public_api