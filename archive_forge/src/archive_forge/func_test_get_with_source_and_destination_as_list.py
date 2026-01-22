from hashlib import md5
from itertools import product
import pytest
from fsspec.implementations.local import make_path_posix
from fsspec.tests.abstract.common import GLOB_EDGE_CASES_TESTS
def test_get_with_source_and_destination_as_list(self, fs, fs_join, local_fs, local_join, local_target, fs_10_files_with_hashed_names):
    source = fs_10_files_with_hashed_names
    target = local_target
    source_files = []
    destination_files = []
    for i in range(10):
        hashed_i = md5(str(i).encode('utf-8')).hexdigest()
        source_files.append(fs_join(source, f'{hashed_i}.txt'))
        destination_files.append(make_path_posix(local_join(target, f'{hashed_i}.txt')))
    fs.get(rpath=source_files, lpath=destination_files)
    for i in range(10):
        file_content = local_fs.cat(destination_files[i]).decode('utf-8')
        assert file_content == str(i)