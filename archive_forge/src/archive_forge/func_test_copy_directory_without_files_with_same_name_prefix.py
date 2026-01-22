from hashlib import md5
from itertools import product
import pytest
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_copy_directory_without_files_with_same_name_prefix(self, fs, fs_join, fs_target, fs_dir_and_file_with_same_name_prefix, supports_empty_directories):
    source = fs_dir_and_file_with_same_name_prefix
    target = fs_target
    fs.cp(fs_join(source, 'subdir'), target, recursive=True)
    assert fs.isfile(fs_join(target, 'subfile.txt'))
    assert not fs.isfile(fs_join(target, 'subdir.txt'))
    fs.rm([fs_join(target, 'subfile.txt')])
    if supports_empty_directories:
        assert fs.ls(target) == []
    else:
        assert not fs.exists(target)
    fs.cp(fs_join(source, 'subdir*'), target, recursive=True)
    assert fs.isdir(fs_join(target, 'subdir'))
    assert fs.isfile(fs_join(target, 'subdir', 'subfile.txt'))
    assert fs.isfile(fs_join(target, 'subdir.txt'))