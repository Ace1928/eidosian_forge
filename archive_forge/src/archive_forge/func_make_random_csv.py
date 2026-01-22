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
def make_random_csv(num_cols=2, num_rows=10, linesep='\r\n', write_names=True):
    arr = np.random.RandomState(42).randint(0, 1000, size=(num_cols, num_rows))
    csv = io.StringIO()
    col_names = list(itertools.islice(generate_col_names(), num_cols))
    if write_names:
        csv.write(','.join(col_names))
        csv.write(linesep)
    for row in arr.T:
        csv.write(','.join(map(str, row)))
        csv.write(linesep)
    csv = csv.getvalue().encode()
    columns = [pa.array(a, type=pa.int64()) for a in arr]
    expected = pa.Table.from_arrays(columns, col_names)
    return (csv, expected)