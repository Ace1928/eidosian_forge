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
def test_array_from_file_reread():
    offset = 9
    fname = 'test.bin'
    with InTemporaryDirectory():
        openers = [open, gzip.open, bz2.BZ2File, BytesIO]
        if HAVE_ZSTD:
            openers += [pyzstd.ZstdFile]
        for shape, opener, dtt, order in itertools.product(((64,), (64, 65), (64, 65, 66)), openers, (np.int16, np.float32), ('F', 'C')):
            n_els = np.prod(shape)
            in_arr = np.arange(n_els, dtype=dtt).reshape(shape)
            is_bio = hasattr(opener, 'getvalue')
            fobj_w = opener() if is_bio else opener(fname, 'wb')
            fobj_w.write(b' ' * offset)
            fobj_w.write(in_arr.tobytes(order=order))
            if is_bio:
                fobj_r = fobj_w
            else:
                fobj_w.close()
                fobj_r = opener(fname, 'rb')
            try:
                out_arr = array_from_file(shape, dtt, fobj_r, offset, order)
                assert_array_equal(in_arr, out_arr)
                out_arr[..., 0] = -1
                assert not np.allclose(in_arr, out_arr)
                out_arr2 = array_from_file(shape, dtt, fobj_r, offset, order)
                assert_array_equal(in_arr, out_arr2)
            finally:
                fobj_r.close()
            del out_arr, out_arr2
            if not is_bio:
                os.unlink(fname)