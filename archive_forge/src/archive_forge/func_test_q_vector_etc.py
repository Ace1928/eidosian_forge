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
@dicom_test
def test_q_vector_etc():
    dw = didw.Wrapper(DATA)
    assert dw.q_vector is None
    assert dw.b_value is None
    assert dw.b_vector is None
    for pos in range(3):
        q_vec = np.zeros((3,))
        q_vec[pos] = 10.0
        dw = didw.Wrapper(DATA)
        dw.q_vector = q_vec
        assert_array_equal(dw.q_vector, q_vec)
        assert dw.b_value == 10
        assert_array_equal(dw.b_vector, q_vec / 10.0)
    dw = didw.Wrapper(DATA)
    dw.q_vector = np.array([0, 0, 1e-06])
    assert dw.b_value == 0
    assert_array_equal(dw.b_vector, np.zeros((3,)))
    sdw = didw.MosaicWrapper(DATA)
    exp_b, exp_g = EXPECTED_PARAMS
    assert_array_almost_equal(sdw.q_vector, exp_b * np.array(exp_g), 5)
    assert_array_almost_equal(sdw.b_value, exp_b)
    assert_array_almost_equal(sdw.b_vector, exp_g)
    sdw = didw.MosaicWrapper(DATA)
    sdw.q_vector = np.array([0, 0, 1e-06])
    assert sdw.b_value == 0
    assert_array_equal(sdw.b_vector, np.zeros((3,)))