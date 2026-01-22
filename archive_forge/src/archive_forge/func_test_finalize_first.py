import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('data', [pd.Series(1, pd.date_range('2000', periods=4)), pd.DataFrame({'A': [1, 1, 1, 1]}, pd.date_range('2000', periods=4))])
def test_finalize_first(data):
    deprecated_msg = 'first is deprecated'
    data.attrs = {'a': 1}
    with tm.assert_produces_warning(FutureWarning, match=deprecated_msg):
        result = data.first('3D')
        assert result.attrs == {'a': 1}