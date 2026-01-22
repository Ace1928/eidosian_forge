import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype
from h5py._hl.files import direct_vfd
from .common import ut, TestCase
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
def test_hitrate_reset(self):
    dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.TestCacheConfig.test_hitrate_reset')
    fn_h5 = os.path.join(dn_tmp, 'test.h5')
    try:
        with File(fn_h5, mode='x') as f:
            hit_rate = f._id.get_mdc_hit_rate()
            f._id.reset_mdc_hit_rate_stats()
            hit_rate = f._id.get_mdc_hit_rate()
            assert hit_rate == 0
    finally:
        shutil.rmtree(dn_tmp)