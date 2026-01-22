import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('result_names', [[1, 2], 'HK', {'2': 2, '3': 3}, 3, 3.0])
def test_invalid_input_result_names(result_names):
    df1 = pd.DataFrame({'col1': ['a', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({'col1': ['c', 'b', 'c'], 'col2': [1.0, 2.0, np.nan], 'col3': [1.0, 2.0, np.nan]})
    with pytest.raises(TypeError, match=f"Passing 'result_names' as a {type(result_names)} is not supported. Provide 'result_names' as a tuple instead."):
        df1.compare(df2, result_names=result_names)