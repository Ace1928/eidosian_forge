import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_date_cast():
    scalar = pa.scalar(datetime.datetime(2012, 1, 1), type=pa.timestamp('us'))
    expected = datetime.date(2012, 1, 1)
    for ty in [pa.date32(), pa.date64()]:
        result = scalar.cast(ty)
        assert result.as_py() == expected