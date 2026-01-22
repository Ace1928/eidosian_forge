from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_put_list_of_files_to_new_directory(self, fs, fs_join, fs_target, local_join, local_bulk_operations_scenario_0):
    source = local_bulk_operations_scenario_0
    target = fs_target
    fs.mkdir(target)
    source_files = [local_join(source, 'file1'), local_join(source, 'file2'), local_join(source, 'subdir', 'subfile1')]
    fs.put(source_files, fs_join(target, 'newdir') + '/')
    assert fs.isdir(fs_join(target, 'newdir'))
    assert fs.isfile(fs_join(target, 'newdir', 'file1'))
    assert fs.isfile(fs_join(target, 'newdir', 'file2'))
    assert fs.isfile(fs_join(target, 'newdir', 'subfile1'))