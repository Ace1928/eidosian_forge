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
def check_copied_files(destination_dir):
    with localfs.open_input_stream(str(destination_dir / 'file1')) as f:
        assert f.read() == b'test1'
    with localfs.open_input_stream(str(destination_dir / 'file2')) as f:
        assert f.read() == b'test2'