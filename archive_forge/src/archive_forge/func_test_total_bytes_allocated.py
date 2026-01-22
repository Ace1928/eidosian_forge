from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_total_bytes_allocated():
    code = 'if 1:\n    import pyarrow as pa\n\n    assert pa.total_allocated_bytes() == 0\n    '
    res = subprocess.run([sys.executable, '-c', code], universal_newlines=True, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        res.check_returncode()
    assert len(res.stderr.splitlines()) == 0