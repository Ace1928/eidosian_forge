from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_directory_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    assert local_fs.isdir(target)
    for source_slash, target_slash in zip([False, True], [False, True]):
        s = fs_join(source, 'subdir')
        if source_slash:
            s += '/'
        t = target + '/' if target_slash else target
        fs.get(s, t)
        assert local_fs.ls(target) == []
        fs.get(s, t, recursive=True)
        if source_slash:
            assert local_fs.isfile(local_join(target, 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subfile2'))
            assert local_fs.isdir(local_join(target, 'nesteddir'))
            assert local_fs.isfile(local_join(target, 'nesteddir', 'nestedfile'))
            assert not local_fs.exists(local_join(target, 'subdir'))
            local_fs.rm([local_join(target, 'subfile1'), local_join(target, 'subfile2'), local_join(target, 'nesteddir')], recursive=True)
        else:
            assert local_fs.isdir(local_join(target, 'subdir'))
            assert local_fs.isfile(local_join(target, 'subdir', 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subdir', 'subfile2'))
            assert local_fs.isdir(local_join(target, 'subdir', 'nesteddir'))
            assert local_fs.isfile(local_join(target, 'subdir', 'nesteddir', 'nestedfile'))
            local_fs.rm(local_join(target, 'subdir'), recursive=True)
        assert local_fs.ls(target) == []
        fs.get(s, t, recursive=True, maxdepth=1)
        if source_slash:
            assert local_fs.isfile(local_join(target, 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subfile2'))
            assert not local_fs.exists(local_join(target, 'nesteddir'))
            assert not local_fs.exists(local_join(target, 'subdir'))
            local_fs.rm([local_join(target, 'subfile1'), local_join(target, 'subfile2')], recursive=True)
        else:
            assert local_fs.isdir(local_join(target, 'subdir'))
            assert local_fs.isfile(local_join(target, 'subdir', 'subfile1'))
            assert local_fs.isfile(local_join(target, 'subdir', 'subfile2'))
            assert not local_fs.exists(local_join(target, 'subdir', 'nesteddir'))
            local_fs.rm(local_join(target, 'subdir'), recursive=True)
        assert local_fs.ls(target) == []