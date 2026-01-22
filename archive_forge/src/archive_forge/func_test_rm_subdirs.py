from __future__ import with_statement
import array
import os
from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises
def test_rm_subdirs(tmpdir):
    sub_path = os.path.join(tmpdir.strpath, 'am', 'stram')
    full_path = os.path.join(sub_path, 'gram')
    mkdirp(os.path.join(full_path))
    rm_subdirs(sub_path)
    assert os.path.exists(sub_path)
    assert not os.path.exists(full_path)