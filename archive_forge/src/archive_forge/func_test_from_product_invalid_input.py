from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('invalid_input', [1, [1], [1, 2], [[1], 2], 'a', ['a'], ['a', 'b'], [['a'], 'b']])
def test_from_product_invalid_input(invalid_input):
    msg = 'Input must be a list / sequence of iterables|Input must be list-like'
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(iterables=invalid_input)