import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('val,precision,expected', [(-117.9998, 3, -118), (117.9998, 3, 118), (117.9998, 2, 118), (0.000123456, 2, 0.00012)])
def test_round_frac(val, precision, expected):
    result = tmod._round_frac(val, precision=precision)
    assert result == expected