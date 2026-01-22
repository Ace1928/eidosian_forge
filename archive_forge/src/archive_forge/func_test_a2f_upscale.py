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
def test_a2f_upscale():
    info = type_info(np.float32)
    arr = np.array([[info['min'], 2 ** 115, info['max']]], dtype=np.float32)
    slope = np.float32(2 ** 121)
    inter = info['min']
    str_io = BytesIO()
    array_to_file(arr, str_io, np.uint8, intercept=inter, divslope=slope, mn=info['min'], mx=info['max'])
    raw = array_from_file(arr.shape, np.uint8, str_io)
    back = apply_read_scaling(raw, slope, inter)
    top = back - arr
    score = np.abs(top / arr)
    assert np.all(score < 10)