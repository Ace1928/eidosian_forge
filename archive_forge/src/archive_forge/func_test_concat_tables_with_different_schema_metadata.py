from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.pandas
def test_concat_tables_with_different_schema_metadata():
    import pandas as pd
    schema = pa.schema([pa.field('a', pa.string()), pa.field('b', pa.string())])
    values = list('abcdefgh')
    df1 = pd.DataFrame({'a': values, 'b': values})
    df2 = pd.DataFrame({'a': [np.nan] * 8, 'b': values})
    table1 = pa.Table.from_pandas(df1, schema=schema, preserve_index=False)
    table2 = pa.Table.from_pandas(df2, schema=schema, preserve_index=False)
    assert table1.schema.equals(table2.schema)
    assert not table1.schema.equals(table2.schema, check_metadata=True)
    table3 = pa.concat_tables([table1, table2])
    assert table1.schema.equals(table3.schema, check_metadata=True)
    assert table2.schema.equals(table3.schema)