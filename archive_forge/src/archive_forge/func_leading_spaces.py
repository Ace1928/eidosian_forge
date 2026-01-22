from __future__ import absolute_import, print_function
from functools import partial
import re
from .compat import text_type, binary_type
def leading_spaces(l):
    idx = 0
    while idx < len(l) and l[idx] == ' ':
        idx += 1
    return idx