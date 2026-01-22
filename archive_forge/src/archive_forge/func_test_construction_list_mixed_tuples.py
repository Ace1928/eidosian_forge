import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('index_vals', [[('A', 1), 'B'], ['B', ('A', 1)]])
def test_construction_list_mixed_tuples(self, index_vals):
    index = Index(index_vals)
    assert isinstance(index, Index)
    assert not isinstance(index, MultiIndex)