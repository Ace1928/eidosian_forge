import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_assign_bad(self):
    df = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    msg = 'assign\\(\\) takes 1 positional argument but 2 were given'
    with pytest.raises(TypeError, match=msg):
        df.assign(lambda x: x.A)
    msg = "'DataFrame' object has no attribute 'C'"
    with pytest.raises(AttributeError, match=msg):
        df.assign(C=df.A, D=df.A + df.C)