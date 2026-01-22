import lazy_import
from modin import pandas as pd  # noqa: E402
def test_dataframe_constructor():
    pd.DataFrame({'col1': [1, 2, 3], 'col2': list('abc')})