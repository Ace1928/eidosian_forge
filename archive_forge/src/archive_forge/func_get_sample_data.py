import collections
import collections.abc
import contextlib
import functools
import gzip
import itertools
import math
import operator
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import types
import weakref
import numpy as np
import matplotlib
from matplotlib import _api, _c_internal_utils
@_api.delete_parameter('3.8', 'np_load', alternative='open(get_sample_data(..., asfileobj=False))')
def get_sample_data(fname, asfileobj=True, *, np_load=True):
    """
    Return a sample data file.  *fname* is a path relative to the
    :file:`mpl-data/sample_data` directory.  If *asfileobj* is `True`
    return a file object, otherwise just a file path.

    Sample data files are stored in the 'mpl-data/sample_data' directory within
    the Matplotlib package.

    If the filename ends in .gz, the file is implicitly ungzipped.  If the
    filename ends with .npy or .npz, and *asfileobj* is `True`, the file is
    loaded with `numpy.load`.
    """
    path = _get_data_path('sample_data', fname)
    if asfileobj:
        suffix = path.suffix.lower()
        if suffix == '.gz':
            return gzip.open(path)
        elif suffix in ['.npy', '.npz']:
            if np_load:
                return np.load(path)
            else:
                return path.open('rb')
        elif suffix in ['.csv', '.xrc', '.txt']:
            return path.open('r')
        else:
            return path.open('rb')
    else:
        return str(path)