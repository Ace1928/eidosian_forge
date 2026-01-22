import datetime
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def test_pyarrow_engine_lines_false(self):
    df = pd.DataFrame({'a': [1, 2, 3]})
    msg = "dtype_backend numpy is invalid, only 'numpy_nullable' and 'pyarrow' are allowed."
    with pytest.raises(ValueError, match=msg):
        df.convert_dtypes(dtype_backend='numpy')