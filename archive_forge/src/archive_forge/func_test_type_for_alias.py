from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_type_for_alias():
    cases = [('i1', pa.int8()), ('int8', pa.int8()), ('i2', pa.int16()), ('int16', pa.int16()), ('i4', pa.int32()), ('int32', pa.int32()), ('i8', pa.int64()), ('int64', pa.int64()), ('u1', pa.uint8()), ('uint8', pa.uint8()), ('u2', pa.uint16()), ('uint16', pa.uint16()), ('u4', pa.uint32()), ('uint32', pa.uint32()), ('u8', pa.uint64()), ('uint64', pa.uint64()), ('f4', pa.float32()), ('float32', pa.float32()), ('f8', pa.float64()), ('float64', pa.float64()), ('date32', pa.date32()), ('date64', pa.date64()), ('string', pa.string()), ('str', pa.string()), ('binary', pa.binary()), ('time32[s]', pa.time32('s')), ('time32[ms]', pa.time32('ms')), ('time64[us]', pa.time64('us')), ('time64[ns]', pa.time64('ns')), ('timestamp[s]', pa.timestamp('s')), ('timestamp[ms]', pa.timestamp('ms')), ('timestamp[us]', pa.timestamp('us')), ('timestamp[ns]', pa.timestamp('ns')), ('duration[s]', pa.duration('s')), ('duration[ms]', pa.duration('ms')), ('duration[us]', pa.duration('us')), ('duration[ns]', pa.duration('ns')), ('month_day_nano_interval', pa.month_day_nano_interval())]
    for val, expected in cases:
        assert pa.type_for_alias(val) == expected