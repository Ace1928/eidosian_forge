import pytest
import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
@pytest.fixture(scope='session')
def struct_vector_func_fixture():
    """
    Register a vector function that returns a struct array
    """

    def pivot(ctx, k, v, c):
        df = pa.RecordBatch.from_arrays([k, v, c], names=['k', 'v', 'c']).to_pandas()
        df_pivot = df.pivot(columns='c', values='v', index='k').reset_index()
        return pa.RecordBatch.from_pandas(df_pivot).to_struct_array()
    func_name = 'y=pivot(x)'
    doc = empty_udf_doc
    pc.register_vector_function(pivot, func_name, doc, {'k': pa.int64(), 'v': pa.float64(), 'c': pa.utf8()}, pa.struct([('k', pa.int64()), ('v1', pa.float64()), ('v2', pa.float64())]))
    return (pivot, func_name)