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
@pytest.fixture
def subtree_s3fs(request, s3fs):
    prefix = 'pyarrow-filesystem/prefix/'
    return dict(fs=SubTreeFileSystem(prefix, s3fs['fs']), pathfn=prefix.__add__, allow_move_dir=False, allow_append_to_file=False)