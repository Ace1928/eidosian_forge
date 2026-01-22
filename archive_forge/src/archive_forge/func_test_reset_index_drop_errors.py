from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_reset_index_drop_errors(self):
    s = Series(range(4))
    with pytest.raises(KeyError, match='does not match index name'):
        s.reset_index('wrong', drop=True)
    with pytest.raises(KeyError, match='does not match index name'):
        s.reset_index('wrong')
    s = Series(range(4), index=MultiIndex.from_product([[1, 2]] * 2))
    with pytest.raises(KeyError, match='not found'):
        s.reset_index('wrong', drop=True)