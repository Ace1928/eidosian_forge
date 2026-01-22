import numpy as np
import pytest
from pandas.errors import SettingWithCopyError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nonunique_assignment_1750(self):
    df = DataFrame([[1, 1, 'x', 'X'], [1, 1, 'y', 'Y'], [1, 2, 'z', 'Z']], columns=list('ABCD'))
    df = df.set_index(['A', 'B'])
    mi = MultiIndex.from_tuples([(1, 1)])
    df.loc[mi, 'C'] = '_'
    assert (df.xs((1, 1))['C'] == '_').all()