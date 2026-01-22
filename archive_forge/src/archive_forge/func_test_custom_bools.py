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
def test_custom_bools(self):
    opts = ConvertOptions(true_values=['T', 'yes'], false_values=['F', 'no'])
    rows = b'a,b,c\nTrue,T,t\nFalse,F,f\nTrue,yes,yes\nFalse,no,no\nN/A,N/A,N/A\n'
    table = self.read_bytes(rows, convert_options=opts)
    schema = pa.schema([('a', pa.string()), ('b', pa.bool_()), ('c', pa.string())])
    assert table.schema == schema
    assert table.to_pydict() == {'a': ['True', 'False', 'True', 'False', 'N/A'], 'b': [True, False, True, False, None], 'c': ['t', 'f', 'yes', 'no', 'N/A']}