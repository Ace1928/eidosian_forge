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
def test_row_number_offset_in_errors(self):

    def format_msg(msg_format, row, *args):
        if self.use_threads:
            row_info = ''
        else:
            row_info = 'Row #{}: '.format(row)
        return msg_format.format(row_info, *args)
    csv, _ = make_random_csv(4, 100, write_names=True)
    read_options = ReadOptions()
    read_options.block_size = len(csv) / 3
    convert_options = ConvertOptions()
    convert_options.column_types = {'a': pa.int32()}
    csv_bad_columns = csv + b'1,2\r\n'
    message_columns = format_msg('{}Expected 4 columns, got 2', 102)
    with pytest.raises(pa.ArrowInvalid, match=message_columns):
        self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
    csv_bad_type = csv + b'a,b,c,d\r\n'
    message_value = format_msg("In CSV column #0: {}CSV conversion error to int32: invalid value 'a'", 102, csv)
    with pytest.raises(pa.ArrowInvalid, match=message_value):
        self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
    long_row = b'this is a long row' * 15 + b',3\r\n'
    csv_bad_columns_long = csv + long_row
    message_long = format_msg('{}Expected 4 columns, got 2: {} ...', 102, long_row[0:96].decode('utf-8'))
    with pytest.raises(pa.ArrowInvalid, match=message_long):
        self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
    read_options.skip_rows_after_names = 47
    with pytest.raises(pa.ArrowInvalid, match=message_columns):
        self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
    with pytest.raises(pa.ArrowInvalid, match=message_value):
        self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
    with pytest.raises(pa.ArrowInvalid, match=message_long):
        self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
    read_options.skip_rows_after_names = 0
    csv, _ = make_random_csv(4, 100, write_names=False)
    read_options.column_names = ['a', 'b', 'c', 'd']
    csv_bad_columns = csv + b'1,2\r\n'
    message_columns = format_msg('{}Expected 4 columns, got 2', 101)
    with pytest.raises(pa.ArrowInvalid, match=message_columns):
        self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
    csv_bad_columns_long = csv + long_row
    message_long = format_msg('{}Expected 4 columns, got 2: {} ...', 101, long_row[0:96].decode('utf-8'))
    with pytest.raises(pa.ArrowInvalid, match=message_long):
        self.read_bytes(csv_bad_columns_long, read_options=read_options, convert_options=convert_options)
    csv_bad_type = csv + b'a,b,c,d\r\n'
    message_value = format_msg("In CSV column #0: {}CSV conversion error to int32: invalid value 'a'", 101)
    message_value = message_value.format(len(csv))
    with pytest.raises(pa.ArrowInvalid, match=message_value):
        self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)
    read_options.skip_rows = 23
    with pytest.raises(pa.ArrowInvalid, match=message_columns):
        self.read_bytes(csv_bad_columns, read_options=read_options, convert_options=convert_options)
    with pytest.raises(pa.ArrowInvalid, match=message_value):
        self.read_bytes(csv_bad_type, read_options=read_options, convert_options=convert_options)