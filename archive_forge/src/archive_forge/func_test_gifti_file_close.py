import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_gifti_file_close(recwarn):
    gii = load(get_test_data('gifti', 'ascii.gii'))
    with InTemporaryDirectory():
        gii.to_filename('test.gii')
    assert not any((isinstance(r.message, ResourceWarning) for r in recwarn))