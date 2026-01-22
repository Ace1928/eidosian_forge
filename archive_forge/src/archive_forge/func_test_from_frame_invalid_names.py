from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names,expected_error_msg', [('bad_input', 'Names should be list-like for a MultiIndex'), (['a', 'b', 'c'], 'Length of names must match number of levels in MultiIndex')])
def test_from_frame_invalid_names(names, expected_error_msg):
    df = pd.DataFrame([['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b']], columns=MultiIndex.from_tuples([('L1', 'x'), ('L2', 'y')]))
    with pytest.raises(ValueError, match=expected_error_msg):
        MultiIndex.from_frame(df, names=names)