import atexit
import contextlib
import functools
import importlib
import inspect
import os
import os.path as op
import re
import shutil
import sys
import tempfile
import unittest
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from subprocess import STDOUT, CalledProcessError, TimeoutExpired, check_output
from unittest import TestCase
import joblib
import numpy as np
import scipy as sp
from numpy.testing import assert_allclose as np_assert_allclose
from numpy.testing import (
import sklearn
from sklearn.utils import (
from sklearn.utils._array_api import _check_array_api_dispatch
from sklearn.utils.fixes import VisibleDeprecationWarning, parse_version, sp_version
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import (
def create_memmap_backed_data(data, mmap_mode='r', return_folder=False):
    """
    Parameters
    ----------
    data
    mmap_mode : str, default='r'
    return_folder :  bool, default=False
    """
    temp_folder = tempfile.mkdtemp(prefix='sklearn_testing_')
    atexit.register(functools.partial(_delete_folder, temp_folder, warn=True))
    filename = op.join(temp_folder, 'data.pkl')
    joblib.dump(data, filename)
    memmap_backed_data = joblib.load(filename, mmap_mode=mmap_mode)
    result = memmap_backed_data if not return_folder else (memmap_backed_data, temp_folder)
    return result