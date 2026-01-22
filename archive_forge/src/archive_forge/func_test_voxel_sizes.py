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
def test_voxel_sizes(self):
    fake_mf = copy(self.MINIMAL_MF)
    MFW = self.WRAPCLASS
    dw = MFW(fake_mf)
    with pytest.raises(didw.WrapperError):
        dw.voxel_sizes
    fake_frame = fake_frames('PixelMeasuresSequence', 'PixelSpacing', [[2.1, 3.2]])[0]
    fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
    with pytest.raises(didw.WrapperError):
        MFW(fake_mf).voxel_sizes
    fake_mf['SpacingBetweenSlices'] = 4.3
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
    fake_frame.PixelMeasuresSequence[0].SliceThickness = 5.4
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
    del fake_mf['SpacingBetweenSlices']
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
    fake_mf['SharedFunctionalGroupsSequence'] = [None]
    with pytest.raises(didw.WrapperError):
        MFW(fake_mf).voxel_sizes
    fake_mf['PerFrameFunctionalGroupsSequence'] = [fake_frame]
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])
    fake_frame = fake_frames('PixelMeasuresSequence', 'PixelSpacing', [[Decimal('2.1'), Decimal('3.2')]])[0]
    fake_mf['SharedFunctionalGroupsSequence'] = [fake_frame]
    fake_mf['SpacingBetweenSlices'] = Decimal('4.3')
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 4.3])
    fake_frame.PixelMeasuresSequence[0].SliceThickness = Decimal('5.4')
    assert_array_equal(MFW(fake_mf).voxel_sizes, [2.1, 3.2, 5.4])