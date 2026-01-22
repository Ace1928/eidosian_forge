import bz2
import functools
import gzip
import itertools
import os
import tempfile
import threading
import time
import warnings
from io import BytesIO
from os.path import exists
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from packaging.version import Version
from nibabel.testing import (
from ..casting import OK_FLOATS, floor_log2, sctypes, shared_range, type_info
from ..openers import BZ2File, ImageOpener, Opener
from ..optpkg import optional_package
from ..tmpdirs import InTemporaryDirectory
from ..volumeutils import (
def test_a2f_non_numeric():
    dt = np.dtype([('f1', 'f'), ('f2', 'i2')])
    arr = np.zeros((2,), dtype=dt)
    arr['f1'] = (0.4, 0.6)
    arr['f2'] = (10, 12)
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, dt)
    assert_array_equal(back_arr, arr)
    try:
        arr.astype(float)
    except (TypeError, ValueError):
        pass
    else:
        back_arr = write_return(arr, fobj, float)
        assert_array_equal(back_arr, arr.astype(float))
    with pytest.raises(ValueError):
        write_return(arr, fobj, float, mn=0)
    with pytest.raises(ValueError):
        write_return(arr, fobj, float, mx=10)