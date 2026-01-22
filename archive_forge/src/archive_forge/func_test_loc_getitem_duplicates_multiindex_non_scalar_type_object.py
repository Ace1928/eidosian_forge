import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_getitem_duplicates_multiindex_non_scalar_type_object():
    df = DataFrame([[np.mean, np.median], ['mean', 'median']], columns=MultiIndex.from_tuples([('functs', 'mean'), ('functs', 'median')]), index=['function', 'name'])
    result = df.loc['function', ('functs', 'mean')]
    expected = np.mean
    assert result == expected