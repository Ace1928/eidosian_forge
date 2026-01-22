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
def test_int_scinter():
    assert int_scinter_ftype(np.int8, 1.0, 0.0) == np.float32
    assert int_scinter_ftype(np.int8, -1.0, 0.0) == np.float32
    assert int_scinter_ftype(np.int8, 1e+38, 0.0) == np.float64
    assert int_scinter_ftype(np.int8, -1e+38, 0.0) == np.float64