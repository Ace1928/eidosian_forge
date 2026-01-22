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
def test_cast_chunked_array_empty():
    for typ1, typ2 in [(pa.dictionary(pa.int8(), pa.string()), pa.string()), (pa.int64(), pa.int32())]:
        arr = pa.chunked_array([], type=typ1)
        result = arr.cast(typ2)
        expected = pa.chunked_array([], type=typ2)
        assert result.equals(expected)