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
def test_a2f_int_scaling():
    arr = np.array([0, 1, 128, 255], dtype=np.uint8)
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.uint8, intercept=1)
    assert_array_equal(back_arr, np.clip(arr - 1.0, 0, 255))
    back_arr = write_return(arr, fobj, np.uint8, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip(arr / 2.0, 0, 255)))
    back_arr = write_return(arr, fobj, np.uint8, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round(np.clip((arr - 1.0) / 2.0, 0, 255)))
    back_arr = write_return(arr, fobj, np.int16, intercept=1, divslope=2)
    assert_array_equal(back_arr, np.round((arr - 1.0) / 2.0))