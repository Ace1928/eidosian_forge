import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from .utils import df_equals, test_data_values
def test_wide_to_long():
    data = pd.DataFrame({'hr1': [514, 573], 'hr2': [545, 526], 'team': ['Red Sox', 'Yankees'], 'year1': [2007, 2008], 'year2': [2008, 2008]})
    with warns_that_defaulting_to_pandas():
        df = pd.wide_to_long(data, ['hr', 'year'], 'team', 'index')
        assert isinstance(df, pd.DataFrame)
    with pytest.raises(ValueError):
        pd.wide_to_long(data.to_numpy(), ['hr', 'year'], 'team', 'index')