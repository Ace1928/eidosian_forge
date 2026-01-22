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
def test_wrapper_args_kwds():
    dcm = didw.wrapper_from_file(DATA_FILE)
    data = dcm.get_data()
    dcm2 = didw.wrapper_from_file(DATA_FILE, np.inf)
    assert_array_equal(data, dcm2.get_data())
    dcm2 = didw.wrapper_from_file(DATA_FILE, defer_size=np.inf)
    assert_array_equal(data, dcm2.get_data())
    csa_fname = pjoin(IO_DATA_PATH, 'csa2_b0.bin')
    with pytest.raises(pydicom.filereader.InvalidDicomError):
        didw.wrapper_from_file(csa_fname)
    dcm_malo = didw.wrapper_from_file(csa_fname, force=True)
    assert not dcm_malo.is_mosaic