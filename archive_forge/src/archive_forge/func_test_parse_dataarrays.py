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
def test_parse_dataarrays():
    fn = 'bad_daa.gii'
    img = gi.GiftiImage()
    with InTemporaryDirectory():
        save(img, fn)
        with open(fn) as fp:
            txt = fp.read()
        txt = txt.replace('NumberOfDataArrays="0"', 'NumberOfDataArrays ="1"')
        with open(fn, 'w') as fp:
            fp.write(txt)
        with clear_and_catch_warnings() as w:
            warnings.filterwarnings('once', category=UserWarning)
            load(fn)
            assert len(w) == 1
            assert img.numDA == 0