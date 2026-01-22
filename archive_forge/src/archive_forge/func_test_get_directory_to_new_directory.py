from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_directory_to_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    for source_slash, target_slash in zip([False, True], [False, True]):
        s = fs_join(source, 'subdir')
        if source_slash:
            s += '/'
        t = local_join(target, 'newdir')
        if target_slash:
            t += '/'
        fs.get(s, t)
        assert local_fs.ls(target) == []
        fs.get(s, t, recursive=True)
        assert local_fs.isdir(local_join(target, 'newdir'))
        assert local_fs.isfile(local_join(target, 'newdir', 'subfile1'))
        assert local_fs.isfile(local_join(target, 'newdir', 'subfile2'))
        assert local_fs.isdir(local_join(target, 'newdir', 'nesteddir'))
        assert local_fs.isfile(local_join(target, 'newdir', 'nesteddir', 'nestedfile'))
        assert not local_fs.exists(local_join(target, 'subdir'))
        local_fs.rm(local_join(target, 'newdir'), recursive=True)
        assert local_fs.ls(target) == []
        fs.get(s, t, recursive=True, maxdepth=1)
        assert local_fs.isdir(local_join(target, 'newdir'))
        assert local_fs.isfile(local_join(target, 'newdir', 'subfile1'))
        assert local_fs.isfile(local_join(target, 'newdir', 'subfile2'))
        assert not local_fs.exists(local_join(target, 'newdir', 'nesteddir'))
        assert not local_fs.exists(local_join(target, 'subdir'))
        local_fs.rm(local_join(target, 'newdir'), recursive=True)
        assert not local_fs.exists(local_join(target, 'newdir'))