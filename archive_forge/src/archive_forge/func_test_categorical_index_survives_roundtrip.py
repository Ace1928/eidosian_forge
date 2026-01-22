import io
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.fs import LocalFileSystem, SubTreeFileSystem
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_categorical_index_survives_roundtrip():
    df = pd.DataFrame([['a', 'b'], ['c', 'd']], columns=['c1', 'c2'])
    df['c1'] = df['c1'].astype('category')
    df = df.set_index(['c1'])
    table = pa.Table.from_pandas(df)
    bos = pa.BufferOutputStream()
    pq.write_table(table, bos)
    ref_df = pq.read_pandas(bos.getvalue()).to_pandas()
    assert isinstance(ref_df.index, pd.CategoricalIndex)
    assert ref_df.index.equals(df.index)