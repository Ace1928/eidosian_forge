import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas._config.config as cf
from pandas import Index
import pandas._testing as tm
def test_format_different_scalar_lengths(self):
    idx = Index(['aaaaaaaaa', 'b'])
    expected = ['aaaaaaaaa', 'b']
    msg = 'Index\\.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert idx.format() == expected