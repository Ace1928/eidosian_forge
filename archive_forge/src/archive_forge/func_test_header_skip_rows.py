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
def test_header_skip_rows(self):
    super().test_header_skip_rows()
    rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
    opts = ReadOptions()
    opts.skip_rows = 4
    opts.column_names = ['ab', 'cd']
    reader = self.open_bytes(rows, read_options=opts)
    with pytest.raises(StopIteration):
        assert reader.read_next_batch()