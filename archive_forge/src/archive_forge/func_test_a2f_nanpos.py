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
def test_a2f_nanpos():
    arr = np.array([np.nan])
    fobj = BytesIO()
    back_arr = write_return(arr, fobj, np.int8, divslope=2)
    assert_array_equal(back_arr, 0)
    back_arr = write_return(arr, fobj, np.int8, intercept=10, divslope=2)
    assert_array_equal(back_arr, -5)