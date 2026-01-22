import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.skipif(sys.platform == 'win32' and (not util.windows_has_tzdata()), reason='Timezone database is not installed on Windows')
def test_timestamp_scalar():
    a = repr(pa.scalar('0000-01-01').cast(pa.timestamp('s')))
    assert a == "<pyarrow.TimestampScalar: '0000-01-01T00:00:00'>"
    b = repr(pa.scalar(datetime.datetime(2015, 1, 1), type=pa.timestamp('s', tz='UTC')))
    assert b == "<pyarrow.TimestampScalar: '2015-01-01T00:00:00+0000'>"
    c = repr(pa.scalar(datetime.datetime(2015, 1, 1), type=pa.timestamp('us')))
    assert c == "<pyarrow.TimestampScalar: '2015-01-01T00:00:00.000000'>"
    d = repr(pc.assume_timezone(pa.scalar('2000-01-01').cast(pa.timestamp('s')), 'America/New_York'))
    assert d == "<pyarrow.TimestampScalar: '2000-01-01T00:00:00-0500'>"