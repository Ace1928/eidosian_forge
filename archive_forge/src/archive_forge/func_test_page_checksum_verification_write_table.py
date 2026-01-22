from collections import OrderedDict
import io
import warnings
from shutil import copytree
import numpy as np
import pytest
import pyarrow as pa
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem, FileSystem
from pyarrow.tests import util
from pyarrow.tests.parquet.common import (_check_roundtrip, _roundtrip_table,
def test_page_checksum_verification_write_table(tempdir):
    """Check that checksum verification works for datasets created with
    pq.write_table()"""
    original_path = tempdir / 'correct.parquet'
    table_orig = pa.table({'a': [1, 2, 3, 4]})
    pq.write_table(table_orig, original_path, write_page_checksum=True)
    table_check = pq.read_table(original_path, page_checksum_verification=True)
    assert table_orig == table_check
    bin_data = bytearray(original_path.read_bytes())
    assert bin_data[31] != bin_data[36]
    bin_data[31], bin_data[36] = (bin_data[36], bin_data[31])
    corrupted_path = tempdir / 'corrupted.parquet'
    corrupted_path.write_bytes(bin_data)
    table_corrupt = pq.read_table(corrupted_path, page_checksum_verification=False)
    assert table_corrupt != table_orig
    assert table_corrupt == pa.table({'a': [1, 3, 2, 4]})
    with pytest.raises(OSError, match='CRC checksum verification'):
        _ = pq.read_table(corrupted_path, page_checksum_verification=True)
    corrupted_pq_file = pq.ParquetFile(corrupted_path, page_checksum_verification=False)
    table_corrupt2 = corrupted_pq_file.read()
    assert table_corrupt2 != table_orig
    assert table_corrupt2 == pa.table({'a': [1, 3, 2, 4]})
    corrupted_pq_file = pq.ParquetFile(corrupted_path, page_checksum_verification=True)
    with pytest.raises(OSError, match='CRC checksum verification'):
        _ = corrupted_pq_file.read()