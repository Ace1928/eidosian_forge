from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_list_of_files_to_existing_directory(self, fs, fs_join, fs_bulk_operations_scenario_0, local_fs, local_join, local_target):
    source = fs_bulk_operations_scenario_0
    target = local_target
    local_fs.mkdir(target)
    source_files = [fs_join(source, 'file1'), fs_join(source, 'file2'), fs_join(source, 'subdir', 'subfile1')]
    for target_slash in [False, True]:
        t = target + '/' if target_slash else target
        fs.get(source_files, t)
        assert local_fs.isfile(local_join(target, 'file1'))
        assert local_fs.isfile(local_join(target, 'file2'))
        assert local_fs.isfile(local_join(target, 'subfile1'))
        local_fs.rm([local_join(target, 'file1'), local_join(target, 'file2'), local_join(target, 'subfile1')], recursive=True)
        assert local_fs.ls(target) == []