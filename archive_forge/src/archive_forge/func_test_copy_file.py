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
def test_copy_file(fs, pathfn):
    s = pathfn('test-copy-source-file')
    t = pathfn('test-copy-target-file')
    with fs.open_output_stream(s):
        pass
    fs.copy_file(s, t)
    fs.delete_file(s)
    fs.delete_file(t)