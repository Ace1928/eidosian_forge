from datetime import (
import operator
import numpy as np
import pytest
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
@pytest.mark.parametrize('td_nat', [NaT, np.timedelta64('NaT', 'ns'), np.timedelta64('NaT')])
@pytest.mark.parametrize('op', [operator.mul, ops.rmul])
def test_td_mul_nat(self, op, td_nat):
    td = Timedelta(10, unit='d')
    typs = '|'.join(['numpy.timedelta64', 'NaTType', 'Timedelta'])
    msg = '|'.join([f"unsupported operand type\\(s\\) for \\*: '{typs}' and '{typs}'", "ufunc '?multiply'? cannot use operands with types"])
    with pytest.raises(TypeError, match=msg):
        op(td, td_nat)