import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture
def target_source(self):
    data = {'A': [0.0, 1.0, 2.0, 3.0, 4.0], 'B': [0.0, 1.0, 0.0, 1.0, 0.0], 'C': ['foo1', 'foo2', 'foo3', 'foo4', 'foo5'], 'D': bdate_range('1/1/2009', periods=5)}
    target = DataFrame(data, index=Index(['a', 'b', 'c', 'd', 'e'], dtype=object))
    source = DataFrame({'MergedA': data['A'], 'MergedD': data['D']}, index=data['C'])
    return (target, source)