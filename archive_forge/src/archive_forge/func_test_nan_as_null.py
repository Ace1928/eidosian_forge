from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
def test_nan_as_null():
    table = pa.table({'a': [1, 2, 3, 4]})
    with pytest.raises(RuntimeError):
        table.__dataframe__(nan_as_null=True)