from __future__ import annotations
import io
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
import pytest
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths, open_files
from s3fs import S3FileSystem as DaskS3FileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
@pytest.mark.parametrize('metadata_file', [True, False])
def test_parquet(s3, engine, s3so, metadata_file):
    dd = pytest.importorskip('dask.dataframe')
    pd = pytest.importorskip('pandas')
    np = pytest.importorskip('numpy')
    url = 's3://%s/test.parquet' % test_bucket_name
    data = pd.DataFrame({'i32': np.arange(1000, dtype=np.int32), 'i64': np.arange(1000, dtype=np.int64), 'f': np.arange(1000, dtype=np.float64), 'bhello': np.random.choice(['hello', 'you', 'people'], size=1000).astype('O')}, index=pd.Index(np.arange(1000), name='foo'))
    df = dd.from_pandas(data, chunksize=500)
    df.to_parquet(url, engine=engine, storage_options=s3so, write_metadata_file=metadata_file)
    files = [f.split('/')[-1] for f in s3.ls(url)]
    if metadata_file:
        assert '_common_metadata' in files
        assert '_metadata' in files
    assert 'part.0.parquet' in files
    df2 = dd.read_parquet(url, index='foo', calculate_divisions=True, engine=engine, storage_options=s3so)
    assert len(df2.divisions) > 1
    dd.utils.assert_eq(data, df2)
    if fsspec_parquet:
        with pytest.raises(ValueError):
            dd.read_parquet(url, engine=engine, storage_options=s3so, open_file_options={'precache_options': {'method': 'parquet', 'engine': 'foo'}}).compute()
        dd.read_parquet(url, engine=engine, storage_options=s3so, open_file_options={'precache_options': {'method': 'parquet', 'max_block': 8000}}).compute()
    fs = get_fs_token_paths(url, storage_options=s3so)[0]

    def _open(*args, check=True, **kwargs):
        assert check
        return fs.open(*args, **kwargs)
    with pytest.raises(AssertionError):
        dd.read_parquet(url, engine=engine, storage_options=s3so, open_file_options={'open_file_func': _open, 'check': False}).compute()
    df3 = dd.read_parquet(url, engine=engine, storage_options=s3so, open_file_options={'open_file_func': _open})
    dd.utils.assert_eq(data, df3)
    df4 = dd.read_parquet(url, engine=engine, storage_options=s3so, open_file_options={'cache_type': 'all'})
    dd.utils.assert_eq(data, df4)