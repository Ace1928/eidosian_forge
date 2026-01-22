import shutil
import sys
import warnings
from os.path import basename, dirname
from os.path import join as pjoin
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from ...loadsave import load, save
from ...nifti1 import xform_codes
from ...testing import clear_and_catch_warnings, suppress_warnings
from ...tmpdirs import InTemporaryDirectory
from .. import gifti as gi
from ..parse_gifti_fast import GiftiImageParser, GiftiParseError
from ..util import gifti_endian_codes
def test_base64_written():
    with InTemporaryDirectory():
        with open(DATA_FILE5, 'rb') as fobj:
            contents = fobj.read()
        assert b'GIFTI_ENCODING_B64BIN' in contents
        assert b'GIFTI_ENDIAN_LITTLE' in contents
        assert b'Base64Binary' not in contents
        assert b'LittleEndian' not in contents
        img5 = load(DATA_FILE5)
        save(img5, 'fixed.gii')
        with open('fixed.gii', 'rb') as fobj:
            contents = fobj.read()
        assert b'GIFTI_ENCODING_B64BIN' not in contents
        assert b'GIFTI_ENDIAN_LITTLE' not in contents
        assert b'Base64Binary' in contents
        if sys.byteorder == 'little':
            assert b'LittleEndian' in contents
        else:
            assert b'BigEndian' in contents
        img5_fixed = load('fixed.gii')
        darrays = img5_fixed.darrays
        assert_array_almost_equal(darrays[0].data, DATA_FILE5_darr1)
        assert_array_almost_equal(darrays[1].data, DATA_FILE5_darr2)