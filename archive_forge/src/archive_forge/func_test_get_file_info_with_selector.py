from datetime import datetime, timezone, timedelta
import gzip
import os
import pathlib
import subprocess
import sys
import pytest
import weakref
import pyarrow as pa
from pyarrow.tests.test_io import assert_file_not_found
from pyarrow.tests.util import (_filesystem_uri, ProxyHandler,
from pyarrow.fs import (FileType, FileInfo, FileSelector, FileSystem,
def test_get_file_info_with_selector(fs, pathfn):
    base_dir = pathfn('selector-dir/')
    file_a = pathfn('selector-dir/test_file_a')
    file_b = pathfn('selector-dir/test_file_b')
    dir_a = pathfn('selector-dir/test_dir_a')
    file_c = pathfn('selector-dir/test_dir_a/test_file_c')
    dir_b = pathfn('selector-dir/test_dir_b')
    try:
        fs.create_dir(base_dir)
        with fs.open_output_stream(file_a):
            pass
        with fs.open_output_stream(file_b):
            pass
        fs.create_dir(dir_a)
        with fs.open_output_stream(file_c):
            pass
        fs.create_dir(dir_b)
        selector = FileSelector(base_dir, allow_not_found=False, recursive=True)
        assert selector.base_dir == base_dir
        infos = fs.get_file_info(selector)
        if fs.type_name == "py::fsspec+('s3', 's3a')":
            len(infos) == 4
        else:
            assert len(infos) == 5
        for info in infos:
            if info.path.endswith(file_a) or info.path.endswith(file_b) or info.path.endswith(file_c):
                assert info.type == FileType.File
            elif info.path.rstrip('/').endswith(dir_a) or info.path.rstrip('/').endswith(dir_b):
                assert info.type == FileType.Directory
            else:
                raise ValueError('unexpected path {}'.format(info.path))
            check_mtime_or_absent(info)
        selector = FileSelector(base_dir, recursive=False)
        infos = fs.get_file_info(selector)
        if fs.type_name == "py::fsspec+('s3', 's3a')":
            assert len(infos) == 3
        else:
            assert len(infos) == 4
    finally:
        fs.delete_dir(base_dir)