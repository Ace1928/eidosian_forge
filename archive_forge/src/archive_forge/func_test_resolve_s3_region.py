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
@pytest.mark.s3
def test_resolve_s3_region():
    from pyarrow.fs import resolve_s3_region
    assert resolve_s3_region('voltrondata-labs-datasets') == 'us-east-2'
    assert resolve_s3_region('mf-nwp-models') == 'eu-west-1'
    with pytest.raises(ValueError, match='Not a valid bucket name'):
        resolve_s3_region('foo/bar')
    with pytest.raises(ValueError, match='Not a valid bucket name'):
        resolve_s3_region('s3:bucket')