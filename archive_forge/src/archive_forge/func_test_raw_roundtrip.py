from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
@pytest.mark.parametrize('data', ['👍...', 'Ωœ∑`...', 'abcd...'])
def test_raw_roundtrip(self, data):
    df = DataFrame({'data': [data]})
    df.to_clipboard()
    result = read_clipboard()
    tm.assert_frame_equal(df, result)