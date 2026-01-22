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
def assert_copy_ok(hdr1, hdr2):
    assert hdr1 is not hdr2
    assert hdr1.permit_truncated == hdr2.permit_truncated
    assert hdr1.general_info is not hdr2.general_info
    assert_arr_dict_equal(hdr1.general_info, hdr2.general_info)
    assert hdr1.image_defs is not hdr2.image_defs
    assert_structarr_equal(hdr1.image_defs, hdr2.image_defs)