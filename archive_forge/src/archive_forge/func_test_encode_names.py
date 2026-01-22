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
def test_encode_names():
    b0 = b'\x00'
    assert encode_value_in_name(0, 'foo', 10) == b'foo' + b0 * 7
    assert encode_value_in_name(1, 'foo', 10) == b'foo' + b0 * 7
    assert encode_value_in_name(8, 'foo', 10) == b'foo' + b0 + b'8' + b0 * 5
    assert encode_value_in_name(40, 'foobar', 10) == b'foobar' + b0 + b'40' + b0
    assert encode_value_in_name(1, 'foobarbazz', 10) == b'foobarbazz'
    with pytest.raises(ValueError):
        encode_value_in_name(1, 'foobarbazzz', 10)
    with pytest.raises(ValueError):
        encode_value_in_name(2, 'foobarbazzz', 10)
    assert encode_value_in_name(2, 'foobarba', 10) == b'foobarba\x002'