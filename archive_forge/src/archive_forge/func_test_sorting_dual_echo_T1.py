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
def test_sorting_dual_echo_T1():
    t1_par = pjoin(DATA_PATH, 'T1_dual_echo.PAR')
    with open(t1_par) as fobj:
        t1_hdr = PARRECHeader.from_fileobj(fobj, strict_sort=True)
    np.random.shuffle(t1_hdr.image_defs)
    sorted_indices = t1_hdr.get_sorted_slice_indices()
    sorted_echos = t1_hdr.image_defs['echo number'][sorted_indices]
    n_half = len(t1_hdr.image_defs) // 2
    assert np.all(sorted_echos[:n_half] == 1)
    assert np.all(sorted_echos[n_half:] == 2)
    vol_labels = t1_hdr.get_volume_labels()
    assert list(vol_labels.keys()) == ['echo number']
    assert_array_equal(vol_labels['echo number'], [1, 2])