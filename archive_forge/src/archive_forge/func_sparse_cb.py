from __future__ import absolute_import, division, print_function
from collections import OrderedDict
from itertools import repeat, chain
import warnings
import numpy as np
from .util import import_
from .core import ODESys, RecoverableError
from .util import (
def sparse_cb(x, y, p=()):
    data = cb(x, y, p).flatten()
    return csc_matrix((data, self._rowvals, self._colptrs))