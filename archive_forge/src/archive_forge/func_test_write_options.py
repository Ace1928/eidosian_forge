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
def test_write_options():
    cls = WriteOptions
    opts = cls()
    check_options_class(cls, include_header=[True, False], delimiter=[',', '\t', '|'], quoting_style=['needed', 'none', 'all_valid'])
    assert opts.batch_size > 0
    opts.batch_size = 12345
    assert opts.batch_size == 12345
    opts = cls(batch_size=9876)
    assert opts.batch_size == 9876
    opts.validate()
    match = 'WriteOptions: batch_size must be at least 1: 0'
    with pytest.raises(pa.ArrowInvalid, match=match):
        opts = cls()
        opts.batch_size = 0
        opts.validate()