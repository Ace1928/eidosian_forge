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
def test_simple_timestamps(self):
    rows = b'a,b,c\n1970,1970-01-01 00:00:00,1970-01-01 00:00:00.123\n1989,1989-07-14 01:00:00,1989-07-14 01:00:00.123456\n'
    table = self.read_bytes(rows)
    schema = pa.schema([('a', pa.int64()), ('b', pa.timestamp('s')), ('c', pa.timestamp('ns'))])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [1970, 1989], 'b': [datetime(1970, 1, 1), datetime(1989, 7, 14, 1)], 'c': [datetime(1970, 1, 1, 0, 0, 0, 123000), datetime(1989, 7, 14, 1, 0, 0, 123456)]}