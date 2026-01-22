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
def test_auto_dict_encode(self):
    opts = ConvertOptions(auto_dict_encode=True)
    rows = 'a,b\nab,1\ncdé,2\ncdé,3\nab,4'.encode()
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.string())), ('b', pa.int64())])
    expected = {'a': ['ab', 'cdé', 'cdé', 'ab'], 'b': [1, 2, 3, 4]}
    assert table.schema == schema
    assert table.to_pydict() == expected
    opts.auto_dict_max_cardinality = 2
    table = self.read_bytes(rows, convert_options=opts)
    assert table.schema == schema
    assert table.to_pydict() == expected
    opts.auto_dict_max_cardinality = 1
    table = self.read_bytes(rows, convert_options=opts)
    assert table.schema == pa.schema([('a', pa.string()), ('b', pa.int64())])
    assert table.to_pydict() == expected
    opts.auto_dict_max_cardinality = 50
    opts.check_utf8 = False
    rows = b'a,b\nab,1\ncd\xff,2\nab,3'
    table = self.read_bytes(rows, convert_options=opts, validate_full=False)
    assert table.schema == schema
    dict_values = table['a'].chunk(0).dictionary
    assert len(dict_values) == 2
    assert dict_values[0].as_py() == 'ab'
    assert dict_values[1].as_buffer() == b'cd\xff'
    opts.check_utf8 = True
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.binary())), ('b', pa.int64())])
    expected = {'a': [b'ab', b'cd\xff', b'ab'], 'b': [1, 2, 3]}
    assert table.schema == schema
    assert table.to_pydict() == expected