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
def test_custom_nulls(self):
    opts = ConvertOptions(null_values=['Xxx', 'Zzz'])
    rows = b'a,b,c,d\nZzz,"Xxx",1,2\nXxx,#N/A,,Zzz\n'
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.null()), ('b', pa.string()), ('c', pa.string()), ('d', pa.int64())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [None, None], 'b': ['Xxx', '#N/A'], 'c': ['1', ''], 'd': [2, None]}
    opts = ConvertOptions(null_values=['Xxx', 'Zzz'], strings_can_be_null=True)
    table = self.read_bytes(rows, convert_options=opts)
    assert table.to_pydict() == {'a': [None, None], 'b': [None, '#N/A'], 'c': ['1', ''], 'd': [2, None]}
    opts.quoted_strings_can_be_null = False
    table = self.read_bytes(rows, convert_options=opts)
    assert table.to_pydict() == {'a': [None, None], 'b': ['Xxx', '#N/A'], 'c': ['1', ''], 'd': [2, None]}
    opts = ConvertOptions(null_values=[])
    rows = b'a,b\n#N/A,\n'
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.string()), ('b', pa.string())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': ['#N/A'], 'b': ['']}