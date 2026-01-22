from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_table_factory_function_args_pandas():
    import pandas as pd
    with pytest.raises(ValueError):
        pa.table(pd.DataFrame({'a': [1, 2, 3]}), names=['a'])
    with pytest.raises(ValueError):
        pa.table(pd.DataFrame({'a': [1, 2, 3]}), metadata={b'foo': b'bar'})
    schema = pa.schema([('a', pa.int32())])
    table = pa.table(pd.DataFrame({'a': [1, 2, 3]}), schema)
    assert table.column('a').type == pa.int32()