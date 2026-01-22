import pandas as pd
def test_index_from_listlike_with_dtype(self, data):
    idx = pd.Index(data, dtype=data.dtype)
    assert idx.dtype == data.dtype
    idx = pd.Index(list(data), dtype=data.dtype)
    assert idx.dtype == data.dtype