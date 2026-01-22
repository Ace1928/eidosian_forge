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
def test_stress_block_sizes(self):
    csv_base, expected = make_random_csv(num_cols=2, num_rows=500)
    block_sizes = [19, 21, 23, 26, 37, 111]
    csvs = [csv_base, csv_base.rstrip(b'\r\n')]
    for csv in csvs:
        for block_size in block_sizes:
            assert csv[:block_size].count(b'\n') >= 2
            read_options = ReadOptions(block_size=block_size)
            reader = self.open_bytes(csv, read_options=read_options)
            table = reader.read_all()
            assert table.schema == expected.schema
            if not table.equals(expected):
                assert table.to_pydict() == expected.to_pydict()