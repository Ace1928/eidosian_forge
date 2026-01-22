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
def test_simple_varied(self):
    rows = b'a,b,c,d\n1,2,3,0\n4.0,-5,foo,True\n'
    table = self.read_bytes(rows)
    schema = pa.schema([('a', pa.float64()), ('b', pa.int64()), ('c', pa.string()), ('d', pa.bool_())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': [1.0, 4.0], 'b': [2, -5], 'c': ['3', 'foo'], 'd': [False, True]}