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
def test_localfs_errors(localfs):
    fs = localfs['fs']
    with assert_file_not_found():
        fs.open_input_stream('/non/existent/file')
    with assert_file_not_found():
        fs.open_output_stream('/non/existent/file')
    with assert_file_not_found():
        fs.create_dir('/non/existent/dir', recursive=False)
    with assert_file_not_found():
        fs.delete_dir('/non/existent/dir')
    with assert_file_not_found():
        fs.delete_file('/non/existent/dir')
    with assert_file_not_found():
        fs.move('/non/existent', '/xxx')
    with assert_file_not_found():
        fs.copy_file('/non/existent', '/xxx')