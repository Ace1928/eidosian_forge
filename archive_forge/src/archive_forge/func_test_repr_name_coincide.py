from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_repr_name_coincide(self):
    index = MultiIndex.from_tuples([('a', 0, 'foo'), ('b', 1, 'bar')], names=['a', 'b', 'c'])
    df = DataFrame({'value': [0, 1]}, index=index)
    lines = repr(df).split('\n')
    assert lines[2].startswith('a 0 foo')