import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
@pytest.mark.memory_leak
@pytest.mark.pandas
def test_deserialize_pandas_arrow_7956():
    df = pd.DataFrame({'a': np.arange(10000), 'b': [test_util.rands(5) for _ in range(10000)]})

    def action():
        df_bytes = pa.ipc.serialize_pandas(df).to_pybytes()
        buf = pa.py_buffer(df_bytes)
        pa.ipc.deserialize_pandas(buf)
    test_util.memory_leak_check(action, threshold=1 << 27, iterations=100)