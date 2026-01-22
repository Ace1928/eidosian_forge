from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
def test_apply_modify_traceback():
    data = DataFrame({'A': ['foo', 'foo', 'foo', 'foo', 'bar', 'bar', 'bar', 'bar', 'foo', 'foo', 'foo'], 'B': ['one', 'one', 'one', 'two', 'one', 'one', 'one', 'two', 'two', 'two', 'one'], 'C': ['dull', 'dull', 'shiny', 'dull', 'dull', 'shiny', 'shiny', 'dull', 'shiny', 'shiny', 'shiny'], 'D': np.random.default_rng(2).standard_normal(11), 'E': np.random.default_rng(2).standard_normal(11), 'F': np.random.default_rng(2).standard_normal(11)})
    data.loc[4, 'C'] = np.nan

    def transform(row):
        if row['C'].startswith('shin') and row['A'] == 'foo':
            row['D'] = 7
        return row
    msg = "'float' object has no attribute 'startswith'"
    with pytest.raises(AttributeError, match=msg):
        data.apply(transform, axis=1)