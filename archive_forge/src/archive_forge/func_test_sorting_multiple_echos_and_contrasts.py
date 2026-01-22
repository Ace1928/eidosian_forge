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
def test_sorting_multiple_echos_and_contrasts():
    t1_par = pjoin(DATA_PATH, 'T1_3echo_mag_real_imag_phase.PAR')
    with open(t1_par) as fobj:
        t1_hdr = PARRECHeader.from_fileobj(fobj, strict_sort=True)
    np.random.shuffle(t1_hdr.image_defs)
    sorted_indices = t1_hdr.get_sorted_slice_indices()
    sorted_slices = t1_hdr.image_defs['slice number'][sorted_indices]
    sorted_echos = t1_hdr.image_defs['echo number'][sorted_indices]
    sorted_types = t1_hdr.image_defs['image_type_mr'][sorted_indices]
    ntotal = len(t1_hdr.image_defs)
    nslices = sorted_slices.max()
    nechos = sorted_echos.max()
    for slice_offset in range(ntotal // nslices):
        istart = slice_offset * nslices
        iend = (slice_offset + 1) * nslices
        assert_array_equal(sorted_slices[istart:iend], np.arange(1, nslices + 1))
        current_echo = slice_offset % nechos + 1
        assert np.all(sorted_echos[istart:iend] == current_echo)
    assert np.all(sorted_types[:ntotal // 4] == 0)
    assert np.all(sorted_types[ntotal // 4:ntotal // 2] == 1)
    assert np.all(sorted_types[ntotal // 2:3 * ntotal // 4] == 2)
    assert np.all(sorted_types[3 * ntotal // 4:ntotal] == 3)
    vol_labels = t1_hdr.get_volume_labels()
    assert list(vol_labels.keys()) == ['echo number', 'image_type_mr']
    assert_array_equal(vol_labels['echo number'], [1, 2, 3] * 4)
    assert_array_equal(vol_labels['image_type_mr'], [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])