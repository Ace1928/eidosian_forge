import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
def test_batch_lifetime(self):
    gc.collect()
    old_allocated = pa.total_allocated_bytes()

    def check_one_batch(reader, expected):
        batch = reader.read_next_batch()
        assert batch.to_pydict() == expected
    rows = b'10,11\n12,13\n14,15\n16,17\n'
    read_options = ReadOptions()
    read_options.column_names = ['a', 'b']
    read_options.block_size = 6
    reader = self.open_bytes(rows, read_options=read_options)
    check_one_batch(reader, {'a': [10], 'b': [11]})
    allocated_after_first_batch = pa.total_allocated_bytes()
    check_one_batch(reader, {'a': [12], 'b': [13]})
    assert pa.total_allocated_bytes() <= allocated_after_first_batch
    check_one_batch(reader, {'a': [14], 'b': [15]})
    assert pa.total_allocated_bytes() <= allocated_after_first_batch
    check_one_batch(reader, {'a': [16], 'b': [17]})
    assert pa.total_allocated_bytes() <= allocated_after_first_batch
    with pytest.raises(StopIteration):
        reader.read_next_batch()
    assert pa.total_allocated_bytes() == old_allocated
    reader = None
    assert pa.total_allocated_bytes() == old_allocated