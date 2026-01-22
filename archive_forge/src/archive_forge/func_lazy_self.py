from __future__ import absolute_import, print_function
from functools import partial
import re
from .compat import text_type, binary_type
def lazy_self():
    return_value = func(*args, **kwargs)
    object.__setattr__(self, 'lazy_self', lambda: return_value)
    return return_value