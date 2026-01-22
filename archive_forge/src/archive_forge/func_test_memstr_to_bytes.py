from __future__ import with_statement
import array
import os
from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises
@parametrize('text,value', [('80G', 80 * 1024 ** 3), ('1.4M', int(1.4 * 1024 ** 2)), ('120M', 120 * 1024 ** 2), ('53K', 53 * 1024)])
def test_memstr_to_bytes(text, value):
    assert memstr_to_bytes(text) == value