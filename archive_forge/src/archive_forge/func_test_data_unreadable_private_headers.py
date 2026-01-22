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
@needs_nibabel_data('nitest-dicom')
def test_data_unreadable_private_headers(self):
    with pytest.warns(UserWarning, match='Error while attempting to read CSA header'):
        dw = didw.wrapper_from_file(DATA_FILE_CT)
    assert dw.image_shape == (512, 571)