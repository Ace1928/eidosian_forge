import copy
import os
import sys
import unittest
from io import BytesIO
from os.path import join as pjoin
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arr_dict_equal, clear_and_catch_warnings, data_path, error_warnings
from .. import trk as trk_module
from ..header import Field
from ..tractogram import Tractogram
from ..tractogram_file import HeaderError, HeaderWarning
from ..trk import (
from .test_tractogram import assert_tractogram_equal
def test_decode_names():
    b0 = b'\x00'
    assert decode_value_from_name(b'') == ('', 0)
    assert decode_value_from_name(b'foo' + b0 * 7) == ('foo', 1)
    assert decode_value_from_name(b'foo\x008' + b0 * 5) == ('foo', 8)
    assert decode_value_from_name(b'foobar\x0010\x00') == ('foobar', 10)
    with pytest.raises(ValueError):
        decode_value_from_name(b'foobar\x0010\x01')
    with pytest.raises(HeaderError):
        decode_value_from_name(b'foo\x0010\x00111')