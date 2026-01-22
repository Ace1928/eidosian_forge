import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
@pytest.mark.s3
def test_write_dataset_s3_put_only(s3_server):
    from pyarrow.fs import S3FileSystem
    host, port, _, _ = s3_server['connection']
    fs = S3FileSystem(access_key='limited', secret_key='limited123', endpoint_override='{}:{}'.format(host, port), scheme='http')
    _configure_s3_limited_user(s3_server, _minio_put_only_policy)
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    part = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    ds.write_dataset(table, 'existing-bucket', filesystem=fs, format='feather', create_dir=False, partitioning=part, existing_data_behavior='overwrite_or_ignore')
    result = ds.dataset('existing-bucket', filesystem=fs, format='ipc', partitioning='hive').to_table()
    assert result.equals(table)
    ds.write_dataset(table, 'existing-bucket', filesystem=fs, format='feather', create_dir=True, partitioning=part, existing_data_behavior='overwrite_or_ignore')
    result = ds.dataset('existing-bucket', filesystem=fs, format='ipc', partitioning='hive').to_table()
    assert result.equals(table)
    with pytest.raises(OSError, match="Bucket 'non-existing-bucket' not found"):
        ds.write_dataset(table, 'non-existing-bucket', filesystem=fs, format='feather', create_dir=True, existing_data_behavior='overwrite_or_ignore')
    fs = S3FileSystem(access_key='limited', secret_key='limited123', endpoint_override='{}:{}'.format(host, port), scheme='http', allow_bucket_creation=True)
    with pytest.raises(OSError, match='Access Denied'):
        ds.write_dataset(table, 'non-existing-bucket', filesystem=fs, format='feather', create_dir=True, existing_data_behavior='overwrite_or_ignore')