from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_time_types():
    t1 = pa.time32('s')
    t2 = pa.time32('ms')
    t3 = pa.time64('us')
    t4 = pa.time64('ns')
    assert t1.unit == 's'
    assert t2.unit == 'ms'
    assert t3.unit == 'us'
    assert t4.unit == 'ns'
    assert str(t1) == 'time32[s]'
    assert str(t4) == 'time64[ns]'
    with pytest.raises(ValueError):
        pa.time32('us')
    with pytest.raises(ValueError):
        pa.time64('s')