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
def test_header_autogenerate_column_names(self):
    rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
    opts = ReadOptions()
    opts.autogenerate_column_names = True
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['f0', 'f1'])
    assert table.to_pydict() == {'f0': ['ab', 'ef', 'ij', 'mn'], 'f1': ['cd', 'gh', 'kl', 'op']}
    opts.skip_rows = 3
    table = self.read_bytes(rows, read_options=opts)
    self.check_names(table, ['f0', 'f1'])
    assert table.to_pydict() == {'f0': ['mn'], 'f1': ['op']}
    opts.skip_rows = 4
    with pytest.raises(pa.ArrowInvalid):
        table = self.read_bytes(rows, read_options=opts)