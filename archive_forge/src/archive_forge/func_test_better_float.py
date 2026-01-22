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
def test_better_float():

    def check_against(f1, f2):
        return f1 if FLOAT_TYPES.index(f1) >= FLOAT_TYPES.index(f2) else f2
    for first in FLOAT_TYPES:
        for other in IUINT_TYPES + sctypes['complex']:
            assert better_float_of(first, other) == first
            assert better_float_of(other, first) == first
            for other2 in IUINT_TYPES + sctypes['complex']:
                assert better_float_of(other, other2) == np.float32
                assert better_float_of(other, other2, np.float64) == np.float64
        for second in FLOAT_TYPES:
            assert better_float_of(first, second) == check_against(first, second)
    assert better_float_of('f4', 'f8', 'f4') == np.float64
    assert better_float_of('i4', 'i8', 'f8') == np.float64