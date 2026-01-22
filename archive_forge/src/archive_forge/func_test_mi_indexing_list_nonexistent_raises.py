import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mi_indexing_list_nonexistent_raises():
    s = Series(range(4), index=MultiIndex.from_product([[1, 2], ['a', 'b']]))
    with pytest.raises(KeyError, match="\\['not' 'found'\\] not in index"):
        s.loc[['not', 'found']]