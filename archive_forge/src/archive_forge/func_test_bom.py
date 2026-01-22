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
def test_bom(self):
    rows = b'\xef\xbb\xbfa,b\n1,2\n'
    expected_data = {'a': [1], 'b': [2]}
    table = self.read_bytes(rows)
    assert table.to_pydict() == expected_data