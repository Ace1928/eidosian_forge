from datetime import datetime as dt
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
import pyarrow.interchange as pi
from pyarrow.interchange.column import (
from pyarrow.interchange.from_dataframe import _from_dataframe
@pytest.mark.pandas
def test_allow_copy_false():
    if Version(pd.__version__) < Version('1.5.0'):
        pytest.skip('__dataframe__ added to pandas in 1.5.0')
    df = pd.DataFrame({'a': [0, 1.0, 2.0]})
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)
    df = pd.DataFrame({'dt': [None, dt(2007, 7, 14), dt(2007, 7, 15)]})
    with pytest.raises(RuntimeError):
        pi.from_dataframe(df, allow_copy=False)