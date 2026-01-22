from __future__ import print_function, division, absolute_import
import ctypes
import operator
from collections import OrderedDict
from math import ceil
from datashader import datashape
import numpy as np
from .internal_utils import IndexCallable, isidentifier
def print_unicode_string(s):
    try:
        return s.decode('unicode_escape').encode('ascii')
    except AttributeError:
        return s