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
def test_column_options(self):
    rows = b'1,2,3\n4,5,6'
    read_options = ReadOptions()
    read_options.column_names = ['d', 'e', 'f']
    reader = self.open_bytes(rows, read_options=read_options)
    expected_schema = pa.schema([('d', pa.int64()), ('e', pa.int64()), ('f', pa.int64())])
    self.check_reader(reader, expected_schema, [{'d': [1, 4], 'e': [2, 5], 'f': [3, 6]}])
    convert_options = ConvertOptions()
    convert_options.include_columns = ['f', 'e']
    reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
    expected_schema = pa.schema([('f', pa.int64()), ('e', pa.int64())])
    self.check_reader(reader, expected_schema, [{'e': [2, 5], 'f': [3, 6]}])
    convert_options.column_types = {'e': pa.string()}
    reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
    expected_schema = pa.schema([('f', pa.int64()), ('e', pa.string())])
    self.check_reader(reader, expected_schema, [{'e': ['2', '5'], 'f': [3, 6]}])
    convert_options.include_columns = ['g', 'f', 'e']
    with pytest.raises(KeyError, match="Column 'g' in include_columns does not exist"):
        reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
    convert_options.include_missing_columns = True
    reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
    expected_schema = pa.schema([('g', pa.null()), ('f', pa.int64()), ('e', pa.string())])
    self.check_reader(reader, expected_schema, [{'g': [None, None], 'e': ['2', '5'], 'f': [3, 6]}])
    convert_options.column_types = {'e': pa.string(), 'g': pa.float64()}
    reader = self.open_bytes(rows, read_options=read_options, convert_options=convert_options)
    expected_schema = pa.schema([('g', pa.float64()), ('f', pa.int64()), ('e', pa.string())])
    self.check_reader(reader, expected_schema, [{'g': [None, None], 'e': ['2', '5'], 'f': [3, 6]}])