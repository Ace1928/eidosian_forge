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
def test_write_dataset_max_rows_per_file(tempdir):
    directory = tempdir / 'ds'
    max_rows_per_file = 10
    max_rows_per_group = 10
    num_of_columns = 2
    num_of_records = 35
    record_batch = _generate_data_and_columns(num_of_columns, num_of_records)
    ds.write_dataset(record_batch, directory, format='parquet', max_rows_per_file=max_rows_per_file, max_rows_per_group=max_rows_per_group)
    files_in_dir = os.listdir(directory)
    expected_partitions = num_of_records // max_rows_per_file + 1
    assert len(files_in_dir) == expected_partitions
    result_row_combination = []
    for _, f_file in enumerate(files_in_dir):
        f_path = directory / str(f_file)
        dataset = ds.dataset(f_path, format='parquet')
        result_row_combination.append(dataset.to_table().shape[0])
    assert expected_partitions == len(result_row_combination)
    assert num_of_records == sum(result_row_combination)
    assert all((file_rowcount <= max_rows_per_file for file_rowcount in result_row_combination))