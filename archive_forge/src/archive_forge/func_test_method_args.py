import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
def test_method_args(self):
    funcs1 = ['argmax', 'argmin', 'sum', 'any', 'all', 'cumsum', 'ptp', 'cumprod', 'prod', 'std', 'var', 'mean', 'round', 'min', 'max', 'argsort', 'sort']
    funcs2 = ['compress', 'take', 'repeat']
    for func in funcs1:
        arr = np.random.rand(8, 7)
        arr2 = arr.copy()
        res1 = getattr(arr, func)()
        res2 = getattr(np, func)(arr2)
        if res1 is None:
            res1 = arr
        if res1.dtype.kind in 'uib':
            assert_((res1 == res2).all(), func)
        else:
            assert_(abs(res1 - res2).max() < 1e-08, func)
    for func in funcs2:
        arr1 = np.random.rand(8, 7)
        arr2 = np.random.rand(8, 7)
        res1 = None
        if func == 'compress':
            arr1 = arr1.ravel()
            res1 = getattr(arr2, func)(arr1)
        else:
            arr2 = (15 * arr2).astype(int).ravel()
        if res1 is None:
            res1 = getattr(arr1, func)(arr2)
        res2 = getattr(np, func)(arr1, arr2)
        assert_(abs(res1 - res2).max() < 1e-08, func)