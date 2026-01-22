from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_with_null(self):
    msg = 'Categorical categories cannot be null'
    with pytest.raises(ValueError, match=msg):
        Categorical([np.nan, 'a', 'b', 'c'], categories=[np.nan, 'a', 'b', 'c'])
    with pytest.raises(ValueError, match=msg):
        Categorical([None, 'a', 'b', 'c'], categories=[None, 'a', 'b', 'c'])
    with pytest.raises(ValueError, match=msg):
        Categorical(DatetimeIndex(['nat', '20160101']), categories=[NaT, Timestamp('20160101')])