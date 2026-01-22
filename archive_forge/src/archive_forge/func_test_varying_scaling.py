from glob import glob
from os.path import basename, dirname
from os.path import join as pjoin
from warnings import simplefilter
import numpy as np
import pytest
from numpy import array as npa
from numpy.testing import assert_almost_equal, assert_array_equal
from .. import load as top_load
from .. import parrec
from ..fileholders import FileHolder
from ..nifti1 import Nifti1Extension, Nifti1Header, Nifti1Image
from ..openers import ImageOpener
from ..parrec import (
from ..testing import assert_arr_dict_equal, clear_and_catch_warnings, suppress_warnings
from ..volumeutils import array_from_file
from . import test_spatialimages as tsi
from .test_arrayproxy import check_mmap
def test_varying_scaling():
    img = PARRECImage.load(VARY_REC)
    rec_shape = (64, 64, 27)
    with open(VARY_REC, 'rb') as fobj:
        arr = array_from_file(rec_shape, '<i2', fobj)
    img_defs = img.header.image_defs
    slopes = img_defs['rescale slope']
    inters = img_defs['rescale intercept']
    sc_slopes = img_defs['scale slope']
    scaled_arr = arr.astype(np.float64)
    for i in range(arr.shape[2]):
        scaled_arr[:, :, i] *= slopes[i]
        scaled_arr[:, :, i] += inters[i]
    assert_almost_equal(np.reshape(scaled_arr, img.shape, order='F'), img.get_fdata(), 9)
    for i in range(arr.shape[2]):
        scaled_arr[:, :, i] /= slopes[i] * sc_slopes[i]
    dv_img = PARRECImage.load(VARY_REC, scaling='fp')
    assert_almost_equal(np.reshape(scaled_arr, img.shape, order='F'), dv_img.get_fdata(), 9)