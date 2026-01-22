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
def str_to_number(value):
    is_neg = False
    if value[:1] == '-':
        is_neg = True
        value = value[1:]
    if len(value) < 2:
        value = int(value, 0)
    elif value[0] == '0':
        literal_type = value[1]
        if literal_type in 'xX':
            value = strip_py2_long_suffix(value)
            value = int(value[2:], 16)
        elif literal_type in 'oO':
            value = int(value[2:], 8)
        elif literal_type in 'bB':
            value = int(value[2:], 2)
        else:
            value = int(value, 8)
    else:
        value = int(value, 0)
    return -value if is_neg else value