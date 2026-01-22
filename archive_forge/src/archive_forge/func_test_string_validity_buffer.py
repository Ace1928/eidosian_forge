from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas.compat import (
from pandas.compat.numpy import np_version_lt1p23
import pandas as pd
import pandas._testing as tm
from pandas.core.interchange.column import PandasColumn
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.from_dataframe import from_dataframe
from pandas.core.interchange.utils import ArrowCTypes
def test_string_validity_buffer() -> None:
    pytest.importorskip('pyarrow', '11.0.0')
    df = pd.DataFrame({'a': ['x']}, dtype='large_string[pyarrow]')
    result = df.__dataframe__().get_column_by_name('a').get_buffers()['validity']
    assert result is None