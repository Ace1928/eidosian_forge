from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
@pytest.mark.slow
def test_repr_mixed_big(self):
    biggie = DataFrame({'A': np.random.randn(200), 'B': tm.makeStringIndex(200)}, index=range(200))
    biggie.loc[:20, 'A'] = np.nan
    biggie.loc[:20, 'B'] = np.nan
    repr(biggie)