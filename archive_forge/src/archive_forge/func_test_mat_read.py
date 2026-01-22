import itertools
import unittest
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ..optpkg import optional_package
from ..casting import sctypes_aliases, shared_range, type_info
from ..spatialimages import HeaderDataError
from ..spm99analyze import HeaderTypeError, Spm99AnalyzeHeader, Spm99AnalyzeImage
from ..testing import (
from ..volumeutils import _dt_min_max, apply_read_scaling
from . import test_analyze
@needs_scipy
def test_mat_read(self):
    img_klass = self.image_class
    arr = np.arange(24, dtype=np.int32).reshape((2, 3, 4))
    aff = np.diag([2, 3, 4, 1])
    img = img_klass(arr, aff)
    fm = img.file_map
    for key, value in fm.items():
        value.fileobj = BytesIO()
    img.to_file_map()
    r_img = img_klass.from_file_map(fm)
    assert_array_equal(r_img.get_fdata(), arr)
    assert_array_equal(r_img.affine, aff)
    mat_fileobj = img.file_map['mat'].fileobj
    from scipy.io import loadmat, savemat
    mat_fileobj.seek(0)
    mats = loadmat(mat_fileobj)
    assert 'M' in mats and 'mat' in mats
    from_111 = np.eye(4)
    from_111[:3, 3] = -1
    to_111 = np.eye(4)
    to_111[:3, 3] = 1
    assert_array_equal(mats['mat'], np.dot(aff, from_111))
    assert img.header.default_x_flip
    flipper = np.diag([-1, 1, 1, 1])
    assert_array_equal(mats['M'], np.dot(aff, np.dot(flipper, from_111)))
    mat_fileobj.seek(0)
    savemat(mat_fileobj, dict(M=np.diag([3, 4, 5, 1]), mat=np.diag([6, 7, 8, 1])))
    r_img = img_klass.from_file_map(fm)
    assert_array_equal(r_img.get_fdata(), arr)
    assert_array_equal(r_img.affine, np.dot(np.diag([6, 7, 8, 1]), to_111))
    mat_fileobj.seek(0)
    mat_fileobj.truncate(0)
    savemat(mat_fileobj, dict(M=np.diag([3, 4, 5, 1])))
    r_img = img_klass.from_file_map(fm)
    assert_array_equal(r_img.get_fdata(), arr)
    assert_array_equal(r_img.affine, np.dot(np.diag([3, 4, 5, 1]), np.dot(flipper, to_111)))