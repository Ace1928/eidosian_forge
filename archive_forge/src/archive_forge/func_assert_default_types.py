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
def assert_default_types(loaded):
    default = loaded.__class__()
    for attr in dir(default):
        with suppress_warnings():
            defaulttype = type(getattr(default, attr))
        if defaulttype is type(None):
            continue
        with suppress_warnings():
            loadedtype = type(getattr(loaded, attr))
        assert loadedtype == defaulttype, f'Type mismatch for attribute: {attr} ({loadedtype} != {defaulttype})'