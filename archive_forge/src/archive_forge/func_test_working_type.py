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
def test_working_type():

    def wt(*args, **kwargs):
        return np.dtype(working_type(*args, **kwargs)).str
    d1 = np.atleast_1d
    for in_type in NUMERIC_TYPES:
        in_ts = np.dtype(in_type).str
        assert wt(in_type) == in_ts
        assert wt(in_type, 1, 0) == in_ts
        assert wt(in_type, 1.0, 0.0) == in_ts
        in_val = d1(in_type(0))
        for slope_type in NUMERIC_TYPES:
            sl_val = slope_type(1)
            assert wt(in_type, sl_val, 0.0) == in_ts
            sl_val = slope_type(2)
            out_val = in_val / d1(sl_val)
            assert wt(in_type, sl_val) == out_val.dtype.str
            for inter_type in NUMERIC_TYPES:
                i_val = inter_type(0)
                assert wt(in_type, 1, i_val) == in_ts
                i_val = inter_type(1)
                out_val = in_val - d1(i_val)
                assert wt(in_type, 1, i_val) == out_val.dtype.str
                out_val = (in_val - d1(i_val)) / d1(sl_val)
                assert wt(in_type, sl_val, i_val) == out_val.dtype.str
    f32s = np.dtype(np.float32).str
    assert wt('f4', 1, 0) == f32s
    assert wt(np.dtype('f4'), 1, 0) == f32s