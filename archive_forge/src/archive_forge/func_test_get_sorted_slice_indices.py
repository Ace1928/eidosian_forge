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
def test_get_sorted_slice_indices():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    n_slices = len(HDR_DEFS)
    assert_array_equal(hdr.get_sorted_slice_indices(), range(n_slices))
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS[::-1])
    assert_array_equal(hdr.get_sorted_slice_indices(), [8, 7, 6, 5, 4, 3, 2, 1, 0] + [17, 16, 15, 14, 13, 12, 11, 10, 9] + [26, 25, 24, 23, 22, 21, 20, 19, 18])
    with clear_and_catch_warnings(modules=[parrec], record=True):
        hdr = PARRECHeader(HDR_INFO, HDR_DEFS[:-1], permit_truncated=True)
    assert_array_equal(hdr.get_sorted_slice_indices(), range(n_slices - 9))
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS[::-1], strict_sort=True)
    assert_array_equal(hdr.get_sorted_slice_indices(), range(n_slices)[::-1])