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
def test_cast_timestamp_to_string():
    pytest.importorskip('pytz')
    import pytz
    dt = datetime.datetime(2000, 1, 1, 0, 0, 0, tzinfo=pytz.utc)
    ts = pa.scalar(dt, type=pa.timestamp('ns', tz='UTC'))
    assert ts.cast(pa.string()) == pa.scalar('2000-01-01 00:00:00.000000000Z')