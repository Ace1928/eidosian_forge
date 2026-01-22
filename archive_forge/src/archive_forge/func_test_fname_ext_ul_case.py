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
def test_fname_ext_ul_case():
    with InTemporaryDirectory():
        with open('afile.TXT', 'w') as fobj:
            fobj.write('Interesting information')
        os_cares_case = not exists('afile.txt')
        with open('bfile.txt', 'w') as fobj:
            fobj.write('More interesting information')
        assert fname_ext_ul_case('nofile.txt') == 'nofile.txt'
        assert fname_ext_ul_case('nofile.TXT') == 'nofile.TXT'
        if os_cares_case:
            assert fname_ext_ul_case('afile.txt') == 'afile.TXT'
            assert fname_ext_ul_case('bfile.TXT') == 'bfile.txt'
        else:
            assert fname_ext_ul_case('afile.txt') == 'afile.txt'
            assert fname_ext_ul_case('bfile.TXT') == 'bfile.TXT'
        assert fname_ext_ul_case('afile.TXT') == 'afile.TXT'
        assert fname_ext_ul_case('bfile.txt') == 'bfile.txt'
        assert fname_ext_ul_case('afile.TxT') == 'afile.TxT'