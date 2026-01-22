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
def test_default_types():
    for fname in datafiles:
        img = load(fname)
        assert_default_types(img)
        assert_default_types(img.meta)
        with pytest.warns(DeprecationWarning):
            for nvpair in img.meta.data:
                assert_default_types(nvpair)
        assert_default_types(img.labeltable)
        for darray in img.darrays:
            assert_default_types(darray)
            assert_default_types(darray.coordsys)
            assert_default_types(darray.meta)
            with pytest.warns(DeprecationWarning):
                for nvpair in darray.meta.data:
                    assert_default_types(nvpair)