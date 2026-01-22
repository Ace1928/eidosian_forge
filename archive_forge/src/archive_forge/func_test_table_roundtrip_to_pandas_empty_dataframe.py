from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_table_roundtrip_to_pandas_empty_dataframe():
    import pandas as pd
    data = pd.DataFrame(index=pd.RangeIndex(0, 10, 1))
    table = pa.table(data)
    result = table.to_pandas()
    assert table.num_rows == 10
    assert data.shape == (10, 0)
    assert result.shape == (10, 0)
    assert result.index.equals(data.index)
    data = pd.DataFrame(index=pd.RangeIndex(0, 10, 3))
    table = pa.table(data)
    result = table.to_pandas()
    assert table.num_rows == 4
    assert data.shape == (4, 0)
    assert result.shape == (4, 0)
    assert result.index.equals(data.index)