import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_melt_non_scalar_var_name_raises(self):
    df = DataFrame(data={'a': [1, 2, 3], 'b': [4, 5, 6]}, index=['11', '22', '33'])
    with pytest.raises(ValueError, match='.* must be a scalar.'):
        df.melt(id_vars=['a'], var_name=[1, 2])