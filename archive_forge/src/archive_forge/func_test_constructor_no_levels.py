from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_constructor_no_levels():
    msg = 'non-zero number of levels/codes'
    with pytest.raises(ValueError, match=msg):
        MultiIndex(levels=[], codes=[])
    msg = 'Must pass both levels and codes'
    with pytest.raises(TypeError, match=msg):
        MultiIndex(levels=[])
    with pytest.raises(TypeError, match=msg):
        MultiIndex(codes=[])