from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_file_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    assert local_fs.isdir(target)
    target_file2 = local_join(target, 'file2')
    target_subfile1 = local_join(target, 'subfile1')
    fs.get(fs_join(source, 'file2'), target)
    assert local_fs.isfile(target_file2)
    fs.get(fs_join(source, 'subdir', 'subfile1'), target)
    assert local_fs.isfile(target_subfile1)
    local_fs.rm([target_file2, target_subfile1])
    assert not local_fs.exists(target_file2)
    assert not local_fs.exists(target_subfile1)
    fs.get(fs_join(source, 'file2'), target + '/')
    assert local_fs.isdir(target)
    assert local_fs.isfile(target_file2)
    fs.get(fs_join(source, 'subdir', 'subfile1'), target + '/')
    assert local_fs.isfile(target_subfile1)