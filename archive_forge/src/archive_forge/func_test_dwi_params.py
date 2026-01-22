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
def test_dwi_params():
    dw = didw.wrapper_from_data(DATA)
    b_matrix = dw.b_matrix
    assert b_matrix.shape == (3, 3)
    q = dw.q_vector
    b = np.sqrt(np.sum(q * q))
    g = q / b
    assert_array_almost_equal(b, EXPECTED_PARAMS[0])
    assert_array_almost_equal(g, EXPECTED_PARAMS[1])