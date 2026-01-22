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
def test_seek_tell_logic():
    bio = BytesIO()
    seek_tell(bio, 10)
    assert bio.tell() == 10

    class BabyBio(BytesIO):

        def seek(self, *args):
            raise OSError()
    bio = BabyBio()
    with pytest.raises(OSError):
        bio.seek(10)
    ZEROB = b'\x00'
    bio.write(ZEROB * 10)
    seek_tell(bio, 10)
    assert bio.tell() == 10
    assert bio.getvalue() == ZEROB * 10
    with pytest.raises(OSError):
        bio.seek(20)
    seek_tell(bio, 20, write0=True)
    assert bio.getvalue() == ZEROB * 20