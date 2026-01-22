from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_glob_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, fs_target, supports_empty_directories):
    source = fs_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    if not supports_empty_directories:
        dummy = fs_join(target, 'dummy')
        fs.touch(dummy)
    assert fs.isdir(target)
    for target_slash in [False, True]:
        t = target + '/' if target_slash else target
        fs.cp(fs_join(source, 'subdir', '*'), t)
        assert fs.isfile(fs_join(target, 'subfile1'))
        assert fs.isfile(fs_join(target, 'subfile2'))
        assert not fs.isdir(fs_join(target, 'nesteddir'))
        assert not fs.exists(fs_join(target, 'nesteddir', 'nestedfile'))
        assert not fs.exists(fs_join(target, 'subdir'))
        fs.rm([fs_join(target, 'subfile1'), fs_join(target, 'subfile2')], recursive=True)
        assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])
        for glob, recursive in zip(['*', '**'], [True, False]):
            fs.cp(fs_join(source, 'subdir', glob), t, recursive=recursive)
            assert fs.isfile(fs_join(target, 'subfile1'))
            assert fs.isfile(fs_join(target, 'subfile2'))
            assert fs.isdir(fs_join(target, 'nesteddir'))
            assert fs.isfile(fs_join(target, 'nesteddir', 'nestedfile'))
            assert not fs.exists(fs_join(target, 'subdir'))
            fs.rm([fs_join(target, 'subfile1'), fs_join(target, 'subfile2'), fs_join(target, 'nesteddir')], recursive=True)
            assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])
            fs.cp(fs_join(source, 'subdir', glob), t, recursive=recursive, maxdepth=1)
            assert fs.isfile(fs_join(target, 'subfile1'))
            assert fs.isfile(fs_join(target, 'subfile2'))
            assert not fs.exists(fs_join(target, 'nesteddir'))
            assert not fs.exists(fs_join(target, 'subdir'))
            fs.rm([fs_join(target, 'subfile1'), fs_join(target, 'subfile2')], recursive=True)
            assert fs.ls(target, detail=False) == ([] if supports_empty_directories else [dummy])