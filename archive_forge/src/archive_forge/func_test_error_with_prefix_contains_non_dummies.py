import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_error_with_prefix_contains_non_dummies(dummies_basic):
    dummies_basic['col2_c'] = dummies_basic['col2_c'].astype(object)
    dummies_basic.loc[2, 'col2_c'] = 'str'
    with pytest.raises(TypeError, match='Passed DataFrame contains non-dummy data'):
        from_dummies(dummies_basic, sep='_')