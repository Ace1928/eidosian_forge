import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('cls', [CategoricalDtype, DatetimeTZDtype, IntervalDtype])
def test_astype_categoricaldtype_class_raises(self, cls):
    df = DataFrame({'A': ['a', 'a', 'b', 'c']})
    xpr = f'Expected an instance of {cls.__name__}'
    with pytest.raises(TypeError, match=xpr):
        df.astype({'A': cls})
    with pytest.raises(TypeError, match=xpr):
        df['A'].astype(cls)