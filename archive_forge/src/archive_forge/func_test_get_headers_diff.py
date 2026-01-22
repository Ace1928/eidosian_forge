from io import StringIO
from os.path import join as pjoin
import numpy as np
import pytest
import nibabel as nib
from nibabel.cmdline.diff import *
from nibabel.cmdline.utils import *
from nibabel.testing import data_path
def test_get_headers_diff():
    fnames = [pjoin(data_path, f) for f in ('standard.nii.gz', 'example4d.nii.gz')]
    actual_difference = get_headers_diff([nib.load(f).header for f in fnames])
    expected_difference = {'regular': [np.asarray(b''), np.asarray(b'r')], 'dim_info': [np.asarray(0, 'uint8'), np.asarray(57, 'uint8')], 'dim': [np.array([3, 4, 5, 7, 1, 1, 1, 1], 'int16'), np.array([4, 128, 96, 24, 2, 1, 1, 1], 'int16')], 'datatype': [np.array(2, 'uint8'), np.array(4, 'uint8')], 'bitpix': [np.array(8, 'uint8'), np.array(16, 'uint8')], 'pixdim': [np.array([1.0, 1.0, 3.0, 2.0, 1.0, 1.0, 1.0, 1.0], 'float32'), np.array([-1.0, 2.0, 2.0, 2.19999909, 2000.0, 1.0, 1.0, 1.0], 'float32')], 'slice_end': [np.array(0, 'uint8'), np.array(23, 'uint8')], 'xyzt_units': [np.array(0, 'uint8'), np.array(10, 'uint8')], 'cal_max': [np.array(0.0, 'float32'), np.asarray(1162.0, 'float32')], 'descrip': [np.array(b'', 'S80'), np.array(b'FSL3.3\x00 v2.25 NIfTI-1 Single file format', 'S80')], 'qform_code': [np.array(0, 'int16'), np.array(1, 'int16')], 'sform_code': [np.array(2, 'int16'), np.array(1, 'int16')], 'quatern_b': [np.array(0.0, 'float32'), np.array(-1.9451068140294884e-26, 'float32')], 'quatern_c': [np.array(0.0, 'float32'), np.array(-0.9967085123062134, 'float32')], 'quatern_d': [np.array(0.0, 'float32'), np.array(-0.0810687392950058, 'float32')], 'qoffset_x': [np.array(0.0, 'float32'), np.array(117.8551025390625, 'float32')], 'qoffset_y': [np.array(0.0, 'float32'), np.array(-35.72294235229492, 'float32')], 'qoffset_z': [np.array(0.0, 'float32'), np.array(-7.248798370361328, 'float32')], 'srow_x': [np.array([1.0, 0.0, 0.0, 0.0], 'float32'), np.array([-2.0, 6.71471565e-19, 9.08102451e-18, 117.855103], 'float32')], 'srow_y': [np.array([0.0, 3.0, 0.0, 0.0], 'float32'), np.array([-6.71471565e-19, 1.97371149, -0.355528235, -35.7229424], 'float32')], 'srow_z': [np.array([0.0, 0.0, 2.0, 0.0], 'float32'), np.array([8.25548089e-18, 0.323207617, 2.17108178, -7.24879837], 'float32')]}
    np.testing.assert_equal(actual_difference, expected_difference)