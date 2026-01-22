from __future__ import absolute_import
import cython
import os
import sys
import re
import io
import codecs
import glob
import shutil
import tempfile
from functools import wraps
from . import __version__ as cython_version
def normalise_float_repr(float_str):
    """
    Generate a 'normalised', simple digits string representation of a float value
    to allow string comparisons.  Examples: '.123', '123.456', '123.'
    """
    str_value = float_str.lower().lstrip('0')
    exp = 0
    if 'E' in str_value or 'e' in str_value:
        str_value, exp = str_value.split('E' if 'E' in str_value else 'e', 1)
        exp = int(exp)
    if '.' in str_value:
        num_int_digits = str_value.index('.')
        str_value = str_value[:num_int_digits] + str_value[num_int_digits + 1:]
    else:
        num_int_digits = len(str_value)
    exp += num_int_digits
    result = (str_value[:exp] + '0' * (exp - len(str_value)) + '.' + '0' * -exp + str_value[exp:]).rstrip('0')
    return result if result != '.' else '.0'