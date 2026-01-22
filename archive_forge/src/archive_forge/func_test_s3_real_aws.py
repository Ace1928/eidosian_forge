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
def test_s3_real_aws():
    from pyarrow.fs import S3FileSystem
    default_region = os.environ.get('PYARROW_TEST_S3_REGION') or 'us-east-1'
    fs = S3FileSystem(anonymous=True)
    assert fs.region == default_region
    fs = S3FileSystem(anonymous=True, region='us-east-2')
    entries = fs.get_file_info(FileSelector('voltrondata-labs-datasets/nyc-taxi'))
    assert len(entries) > 0
    key = 'voltrondata-labs-datasets/nyc-taxi/year=2019/month=6/part-0.parquet'
    with fs.open_input_stream(key) as f:
        md = f.metadata()
        assert 'Content-Type' in md
        assert md['Last-Modified'] == b'2022-07-12T23:32:00Z'
        assert md['ETag'] == b'"4c6a76826a695c6ac61592bc30cda3df-16"'