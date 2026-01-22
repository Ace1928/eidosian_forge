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
@pytest.mark.dataset
def test_checksum_write_to_dataset(tempdir):
    """Check that checksum verification works for datasets created with
    pq.write_to_dataset"""
    table_orig = pa.table({'a': [1, 2, 3, 4]})
    original_dir_path = tempdir / 'correct_dir'
    pq.write_to_dataset(table_orig, original_dir_path, write_page_checksum=True)
    original_file_path_list = list(original_dir_path.iterdir())
    assert len(original_file_path_list) == 1
    original_path = original_file_path_list[0]
    table_check = pq.read_table(original_path, page_checksum_verification=True)
    assert table_orig == table_check
    bin_data = bytearray(original_path.read_bytes())
    assert bin_data[31] != bin_data[36]
    bin_data[31], bin_data[36] = (bin_data[36], bin_data[31])
    corrupted_dir_path = tempdir / 'corrupted_dir'
    copytree(original_dir_path, corrupted_dir_path)
    corrupted_file_path = corrupted_dir_path / original_path.name
    corrupted_file_path.write_bytes(bin_data)
    table_corrupt = pq.read_table(corrupted_file_path, page_checksum_verification=False)
    assert table_corrupt != table_orig
    assert table_corrupt == pa.table({'a': [1, 3, 2, 4]})
    with pytest.raises(OSError, match='CRC checksum verification'):
        _ = pq.read_table(corrupted_file_path, page_checksum_verification=True)