import tempfile
import shutil
import os
import numpy as np
from h5py import File, special_dtype
from h5py._hl.files import direct_vfd
from .common import ut, TestCase
from .common import ut, TestCase, UNICODE_FILENAMES, closed_tempfile
def test_mdc_config_get(self):
    dn_tmp = tempfile.mkdtemp('h5py.lowtest.test_h5f.TestFileID.TestCacheConfig.test_mdc_config_get')
    fn_h5 = os.path.join(dn_tmp, 'test.h5')
    try:
        with File(fn_h5, mode='x') as f:
            conf = f._id.get_mdc_config()
            f._id.set_mdc_config(conf)
    finally:
        shutil.rmtree(dn_tmp)