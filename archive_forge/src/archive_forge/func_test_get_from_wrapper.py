import gzip
from copy import copy
from decimal import Decimal
from hashlib import sha1
from os.path import dirname
from os.path import join as pjoin
from unittest import TestCase
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from ...volumeutils import endian_codes
from .. import dicomreaders as didr
from .. import dicomwrappers as didw
from . import dicom_test, have_dicom, pydicom
def test_get_from_wrapper():
    dcm_data = {'some_key': 'some value'}
    dw = didw.Wrapper(dcm_data)
    assert dw.get('some_key') == 'some value'
    assert dw.get('some_other_key') is None
    assert dw['some_key'] == 'some value'
    with pytest.raises(KeyError):
        dw['some_other_key']

    class FakeData(dict):
        pass
    d = FakeData()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert dw.get('some_key') is None

    class FakeData2:

        def get(self, key, default):
            return 1
    d = FakeData2()
    d.some_key = 'another bit of data'
    dw = didw.Wrapper(d)
    assert dw.get('some_key') == 1