import operator
import re
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('attr', ['days', 'seconds', 'microseconds', 'nanoseconds', 'components'])
def test_timedelta_property(attr):
    s = pd.Series(pd.timedelta_range('2000', periods=4))
    s.attrs = {'a': 1}
    result = getattr(s.dt, attr)
    assert result.attrs == {'a': 1}