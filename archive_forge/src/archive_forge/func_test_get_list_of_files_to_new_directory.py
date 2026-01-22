from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_list_of_files_to_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    source_files = [fs_join(source, 'file1'), fs_join(source, 'file2'), fs_join(source, 'subdir', 'subfile1')]
    fs.get(source_files, local_join(target, 'newdir') + '/')
    assert local_fs.isdir(local_join(target, 'newdir'))
    assert local_fs.isfile(local_join(target, 'newdir', 'file1'))
    assert local_fs.isfile(local_join(target, 'newdir', 'file2'))
    assert local_fs.isfile(local_join(target, 'newdir', 'subfile1'))