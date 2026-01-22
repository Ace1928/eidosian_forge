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
def test_file_object(self):
    data = b'a,b\n1,2\n3,4\n'
    expected_data = {'a': [1, 3], 'b': [2, 4]}
    bio = io.BytesIO(data)
    reader = self.open_csv(bio)
    expected_schema = pa.schema([('a', pa.int64()), ('b', pa.int64())])
    self.check_reader(reader, expected_schema, [expected_data])