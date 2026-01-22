import os
import re
import functools
import itertools
import warnings
import weakref
import contextlib
import operator
from operator import itemgetter, index as opindex, methodcaller
from collections.abc import Mapping
import numpy as np
from . import format
from ._datasource import DataSource
from numpy.core import overrides
from numpy.core.multiarray import packbits, unpackbits
from numpy.core._multiarray_umath import _load_from_filelike
from numpy.core.overrides import set_array_function_like_doc, set_module
from ._iotools import (
from numpy.compat import (
def encode_unicode_cols(row_tup):
    row = list(row_tup)
    for i in strcolidx:
        row[i] = row[i].encode('latin1')
    return tuple(row)