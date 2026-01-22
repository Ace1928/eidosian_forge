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
def test_load_getbyintent():
    img = load(DATA_FILE1)
    da = img.get_arrays_from_intent('NIFTI_INTENT_POINTSET')
    assert len(da) == 1
    da = img.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')
    assert len(da) == 1
    da = img.get_arrays_from_intent('NIFTI_INTENT_CORREL')
    assert len(da) == 0
    assert da == []