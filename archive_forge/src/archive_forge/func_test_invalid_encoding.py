from textwrap import dedent
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.io.clipboard import (
def test_invalid_encoding(self, df):
    msg = 'clipboard only supports utf-8 encoding'
    with pytest.raises(ValueError, match=msg):
        df.to_clipboard(encoding='ascii')
    with pytest.raises(NotImplementedError, match=msg):
        read_clipboard(encoding='ascii')