import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_raise_of_column_name_value(self):
    df = DataFrame({'col': list('ABC'), 'value': range(10, 16, 2)})
    with pytest.raises(ValueError, match=re.escape('value_name (value) cannot match')):
        df.melt(id_vars='value', value_name='value')