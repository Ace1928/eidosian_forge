import numpy as np
from pandas import (
import pandas._testing as tm
def test_append_join_nondatetimeindex(self):
    rng = timedelta_range('1 days', periods=10)
    idx = Index(['a', 'b', 'c', 'd'])
    result = rng.append(idx)
    assert isinstance(result[0], Timedelta)
    rng.join(idx, how='outer')