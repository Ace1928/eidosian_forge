from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_file_to_new_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    fs.get(fs_join(source, 'subdir', 'subfile1'), local_join(target, 'newdir/'))
    assert local_fs.isdir(target)
    assert local_fs.isdir(local_join(target, 'newdir'))
    assert local_fs.isfile(local_join(target, 'newdir', 'subfile1'))