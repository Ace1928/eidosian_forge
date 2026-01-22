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
def test__is_compressed_fobj():
    with InTemporaryDirectory():
        file_openers = [('', open, False), ('.gz', gzip.open, True), ('.bz2', BZ2File, True)]
        if HAVE_ZSTD:
            file_openers += [('.zst', pyzstd.ZstdFile, True)]
        for ext, opener, compressed in file_openers:
            fname = 'test.bin' + ext
            for mode in ('wb', 'rb'):
                fobj = opener(fname, mode)
                assert _is_compressed_fobj(fobj) == compressed
                fobj.close()