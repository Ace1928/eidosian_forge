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
def test_iop(self):
    fake_mf = copy(self.MINIMAL_MF)
    MFW = self.WRAPCLASS
    dw = MFW(fake_mf)
    with pytest.raises(didw.WrapperError):
        dw.image_orient_patient
    fake_frame = fake_frames('PlaneOrientationSequence', 'ImageOrientationPatient', [[0, 1, 0, 1, 0, 0]])[0]
    fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
    assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])
    fake_mf['SharedFunctionalGroupsSequence'] = [None]
    with pytest.raises(didw.WrapperError):
        MFW(fake_mf).image_orient_patient
    fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
    assert_array_equal(MFW(fake_mf).image_orient_patient, [[0, 1], [1, 0], [0, 0]])