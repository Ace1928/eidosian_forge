from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_glob_to_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    for target_slash in [False, True]:
        t = fs_join(target, 'newdir')
        if target_slash:
            t += '/'
        fs.cp(fs_join(source, 'subdir', '*'), t)
        assert fs.isdir(fs_join(target, 'newdir'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))
        assert fs.isfile(fs_join(target, 'newdir', 'subfile2'))
        assert not fs.exists(fs_join(target, 'newdir', 'nesteddir'))
        assert not fs.exists(fs_join(target, 'newdir', 'nesteddir', 'nestedfile'))
        assert not fs.exists(fs_join(target, 'subdir'))
        assert not fs.exists(fs_join(target, 'newdir', 'subdir'))
        fs.rm(fs_join(target, 'newdir'), recursive=True)
        assert not fs.exists(fs_join(target, 'newdir'))
        for glob, recursive in zip(['*', '**'], [True, False]):
            fs.cp(fs_join(source, 'subdir', glob), t, recursive=recursive)
            assert fs.isdir(fs_join(target, 'newdir'))
            assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))
            assert fs.isfile(fs_join(target, 'newdir', 'subfile2'))
            assert fs.isdir(fs_join(target, 'newdir', 'nesteddir'))
            assert fs.isfile(fs_join(target, 'newdir', 'nesteddir', 'nestedfile'))
            assert not fs.exists(fs_join(target, 'subdir'))
            assert not fs.exists(fs_join(target, 'newdir', 'subdir'))
            fs.rm(fs_join(target, 'newdir'), recursive=True)
            assert not fs.exists(fs_join(target, 'newdir'))
            fs.cp(fs_join(source, 'subdir', glob), t, recursive=recursive, maxdepth=1)
            assert fs.isdir(fs_join(target, 'newdir'))
            assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))
            assert fs.isfile(fs_join(target, 'newdir', 'subfile2'))
            assert not fs.exists(fs_join(target, 'newdir', 'nesteddir'))
            assert not fs.exists(fs_join(target, 'subdir'))
            assert not fs.exists(fs_join(target, 'newdir', 'subdir'))
            fs.rm(fs_join(target, 'newdir'), recursive=True)
            assert not fs.exists(fs_join(target, 'newdir'))