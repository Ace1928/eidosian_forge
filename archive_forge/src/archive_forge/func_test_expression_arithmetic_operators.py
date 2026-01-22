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
def test_expression_arithmetic_operators():
    dataset = ds.dataset(pa.table({'a': [1, 2, 3], 'b': [2, 2, 2]}))
    a = ds.field('a')
    b = ds.field('b')
    result = dataset.to_table(columns={'a+1': a + 1, 'b-a': b - a, 'a*2': a * 2, 'a/b': a.cast('float64') / b})
    expected = pa.table({'a+1': [2, 3, 4], 'b-a': [1, 0, -1], 'a*2': [2, 4, 6], 'a/b': [0.5, 1.0, 1.5]})
    assert result.equals(expected)