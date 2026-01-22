from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_put_file_to_existing_directory(self, fs, fs_join, fs_target, local_join, local_bulk_operations_scenario_0, supports_empty_directories):
    source = local_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    if not supports_empty_directories:
        fs.touch(fs_join(target, 'dummy'))
    assert fs.isdir(target)
    target_file2 = fs_join(target, 'file2')
    target_subfile1 = fs_join(target, 'subfile1')
    fs.put(local_join(source, 'file2'), target)
    assert fs.isfile(target_file2)
    fs.put(local_join(source, 'subdir', 'subfile1'), target)
    assert fs.isfile(target_subfile1)
    fs.rm([target_file2, target_subfile1])
    assert not fs.exists(target_file2)
    assert not fs.exists(target_subfile1)
    fs.put(local_join(source, 'file2'), target + '/')
    assert fs.isdir(target)
    assert fs.isfile(target_file2)
    fs.put(local_join(source, 'subdir', 'subfile1'), target + '/')
    assert fs.isfile(target_subfile1)