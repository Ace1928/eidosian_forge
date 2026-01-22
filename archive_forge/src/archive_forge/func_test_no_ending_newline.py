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
def test_no_ending_newline(self):
    rows = b'a,b,c\n1,2,3\n4,5,6'
    reader = self.open_bytes(rows)
    expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64()), ('c', pa.int64())])
    self.check_reader(reader, expected_schema, [{'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}])