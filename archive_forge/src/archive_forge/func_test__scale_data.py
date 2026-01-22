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
def test__scale_data(self):
    fake_mf = copy(self.MINIMAL_MF)
    MFW = self.WRAPCLASS
    dw = MFW(fake_mf)
    data = np.arange(24).reshape((2, 3, 4))
    assert_array_equal(data, dw._scale_data(data))
    fake_mf['RescaleSlope'] = 2.0
    fake_mf['RescaleIntercept'] = -1.0
    assert_array_equal(data * 2 - 1, dw._scale_data(data))
    fake_frame = fake_frames('PixelValueTransformationSequence', 'RescaleSlope', [3.0])[0]
    fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
    dw = MFW(fake_mf)
    with pytest.raises(AttributeError):
        dw._scale_data(data)
    fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = -2
    assert_array_equal(data * 3 - 2, dw._scale_data(data))
    fake_frame.PixelValueTransformationSequence[0].RescaleSlope = Decimal('3')
    fake_frame.PixelValueTransformationSequence[0].RescaleIntercept = Decimal('-2')
    assert_array_equal(data * 3 - 2, dw._scale_data(data))