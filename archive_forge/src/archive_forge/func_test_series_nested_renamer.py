from itertools import chain
import re
import numpy as np
import pytest
from pandas.errors import SpecificationError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('renamer', [{'foo': ['min', 'max']}, {'foo': ['min', 'max'], 'bar': ['sum', 'mean']}])
def test_series_nested_renamer(renamer):
    s = Series(range(6), dtype='int64', name='series')
    msg = 'nested renamer is not supported'
    with pytest.raises(SpecificationError, match=msg):
        s.agg(renamer)