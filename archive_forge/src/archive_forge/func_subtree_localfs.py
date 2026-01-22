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
def subtree_localfs(request, tempdir, localfs):
    return dict(fs=SubTreeFileSystem(str(tempdir), localfs['fs']), pathfn=lambda p: p, allow_move_dir=True, allow_append_to_file=True)