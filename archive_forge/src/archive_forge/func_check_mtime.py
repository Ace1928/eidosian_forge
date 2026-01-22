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
def check_mtime(file_info):
    assert isinstance(file_info.mtime, datetime)
    assert isinstance(file_info.mtime_ns, int)
    assert file_info.mtime_ns >= 0
    assert file_info.mtime_ns == pytest.approx(file_info.mtime.timestamp() * 1000000000.0)
    tzinfo = file_info.mtime.tzinfo
    assert tzinfo is not None
    assert tzinfo.utcoffset(None) == timedelta(0)