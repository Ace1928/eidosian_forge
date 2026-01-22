import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hashable_tuple_args():
    df3 = DataFrame({'data': [(1, []), (2, {})]})
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        hash_pandas_object(df3)