import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
@pytest.mark.nopandas
def test_timestamp_nanos_nopandas():
    pytest.importorskip('pytz')
    import pytz
    tz = 'America/New_York'
    ty = pa.timestamp('ns', tz=tz)
    s = pa.scalar(946684800000000000 + 1000, type=ty)
    tzinfo = pytz.timezone(tz)
    expected = datetime.datetime(2000, 1, 1, microsecond=1, tzinfo=tzinfo)
    expected = tzinfo.fromutc(expected)
    result = s.as_py()
    assert result == expected
    assert result.year == 1999
    assert result.hour == 19
    s = pa.scalar(946684800000000001, type=ty)
    with pytest.raises(ValueError):
        s.as_py()