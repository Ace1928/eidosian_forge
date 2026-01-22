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
def test_apply_read_scaling_nones():
    arr = np.arange(10, dtype=np.int16)
    assert_array_equal(apply_read_scaling(arr, None, None), arr)
    assert_array_equal(apply_read_scaling(arr, 2, None), arr * 2)
    assert_array_equal(apply_read_scaling(arr, None, 1), arr + 1)