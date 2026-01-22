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
def test_copy_on_init():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    assert hdr.general_info is not HDR_INFO
    hdr.general_info['max_slices'] = 10
    assert hdr.general_info['max_slices'] == 10
    assert HDR_INFO['max_slices'] == 9
    assert hdr.image_defs is not HDR_DEFS
    hdr.image_defs['image pixel size'] = 8
    assert_array_equal(hdr.image_defs['image pixel size'], 8)
    assert_array_equal(HDR_DEFS['image pixel size'], 16)