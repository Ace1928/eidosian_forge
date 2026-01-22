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
def test_header_scaling():
    hdr = PARRECHeader(HDR_INFO, HDR_DEFS)
    def_scaling = [np.unique(x) for x in hdr.get_data_scaling()]
    fp_scaling = [np.unique(x) for x in hdr.get_data_scaling('fp')]
    dv_scaling = [np.unique(x) for x in hdr.get_data_scaling('dv')]
    assert_array_equal(def_scaling, dv_scaling)
    assert_almost_equal(dv_scaling, [[1.2903541326522827], [0.0]], 5)
    for hdr in (hdr, PARRECHeader(HDR_INFO, HDR_DEFS)):
        scaling = [np.unique(x) for x in hdr.get_data_scaling()]
        assert_array_equal(scaling, dv_scaling)
    assert not np.all(fp_scaling == dv_scaling)