import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
@pytest.mark.parametrize('op', ['+', '-', '*', '/'])
def test_invalid_type_for_operator_raises(self, parser, engine, op):
    df = DataFrame({'a': [1, 2], 'b': ['c', 'd']})
    msg = "unsupported operand type\\(s\\) for .+: '.+' and '.+'|Cannot"
    with pytest.raises(TypeError, match=msg):
        df.eval(f'a {op} b', engine=engine, parser=parser)