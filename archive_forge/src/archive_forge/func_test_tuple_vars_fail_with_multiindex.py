import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('id_vars, value_vars', [[('A', 'a'), [('B', 'b')]], [[('A', 'a')], ('B', 'b')], [('A', 'a'), ('B', 'b')]])
def test_tuple_vars_fail_with_multiindex(self, id_vars, value_vars, df1):
    msg = '(id|value)_vars must be a list of tuples when columns are a MultiIndex'
    with pytest.raises(ValueError, match=msg):
        df1.melt(id_vars=id_vars, value_vars=value_vars)