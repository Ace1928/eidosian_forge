from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('encoding', ['utf-8', 'cp1251'])
def test_to_csv_compression_encoding_gcs(gcs_buffer, compression_only, encoding, compression_to_extension):
    """
    Compression and encoding should with GCS.

    GH 35677 (to_csv, compression), GH 26124 (to_csv, encoding), and
    GH 32392 (read_csv, encoding)
    """
    df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
    compression = {'method': compression_only}
    if compression_only == 'gzip':
        compression['mtime'] = 1
    buffer = BytesIO()
    df.to_csv(buffer, compression=compression, encoding=encoding, mode='wb')
    path_gcs = 'gs://test/test.csv'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)
    read_df = read_csv(path_gcs, index_col=0, compression=compression_only, encoding=encoding)
    tm.assert_frame_equal(df, read_df)
    file_ext = compression_to_extension[compression_only]
    compression['method'] = 'infer'
    path_gcs += f'.{file_ext}'
    df.to_csv(path_gcs, compression=compression, encoding=encoding)
    res = gcs_buffer.getvalue()
    expected = buffer.getvalue()
    assert_equal_zip_safe(res, expected, compression_only)
    read_df = read_csv(path_gcs, index_col=0, compression='infer', encoding=encoding)
    tm.assert_frame_equal(df, read_df)