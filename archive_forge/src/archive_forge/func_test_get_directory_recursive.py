from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_directory_recursive(self, fs, fs_join, fs_path, local_fs, local_join, local_target):
    src = fs_join(fs_path, 'src')
    src_file = fs_join(src, 'file')
    fs.mkdir(src)
    fs.touch(src_file)
    target = local_target
    assert not local_fs.exists(target)
    for loop in range(2):
        fs.get(src, target, recursive=True)
        assert local_fs.isdir(target)
        if loop == 0:
            assert local_fs.isfile(local_join(target, 'file'))
            assert not local_fs.exists(local_join(target, 'src'))
        else:
            assert local_fs.isfile(local_join(target, 'file'))
            assert local_fs.isdir(local_join(target, 'src'))
            assert local_fs.isfile(local_join(target, 'src', 'file'))
    local_fs.rm(target, recursive=True)
    assert not local_fs.exists(target)
    for loop in range(2):
        fs.get(src + '/', target, recursive=True)
        assert local_fs.isdir(target)
        assert local_fs.isfile(local_join(target, 'file'))
        assert not local_fs.exists(local_join(target, 'src'))