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
def test_write_quoting_style():
    t = pa.Table.from_arrays([[1, 2, None], ['a', None, 'c']], ['c1', 'c2'])
    buf = io.BytesIO()
    for write_options, res in [(WriteOptions(quoting_style='none'), b'"c1","c2"\n1,a\n2,\n,c\n'), (WriteOptions(), b'"c1","c2"\n1,"a"\n2,\n,"c"\n'), (WriteOptions(quoting_style='all_valid'), b'"c1","c2"\n"1","a"\n"2",\n,"c"\n')]:
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            writer.write_table(t)
        assert buf.getvalue() == res
        buf.seek(0)
    t = pa.Table.from_arrays([[',', '"']], ['c1'])
    buf = io.BytesIO()
    for write_options, res in [(WriteOptions(quoting_style='needed'), b'"c1"\n","\n""""\n'), (WriteOptions(quoting_style='none'), pa.lib.ArrowInvalid)]:
        with CSVWriter(buf, t.schema, write_options=write_options) as writer:
            try:
                writer.write_table(t)
            except Exception as e:
                assert isinstance(e, res)
                break
        assert buf.getvalue() == res
        buf.seek(0)