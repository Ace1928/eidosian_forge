import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_metadata_immutable(idx):
    levels, codes = (idx.levels, idx.codes)
    mutable_regex = re.compile('does not support mutable operations')
    with pytest.raises(TypeError, match=mutable_regex):
        levels[0] = levels[0]
    with pytest.raises(TypeError, match=mutable_regex):
        levels[0][0] = levels[0][0]
    with pytest.raises(TypeError, match=mutable_regex):
        codes[0] = codes[0]
    with pytest.raises(ValueError, match='assignment destination is read-only'):
        codes[0][0] = codes[0][0]
    names = idx.names
    with pytest.raises(TypeError, match=mutable_regex):
        names[0] = names[0]