import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_reindex_categorical_non_unique_unused_category(self):
    msg = 'cannot reindex on an axis with duplicate labels'
    ci = CategoricalIndex(['a', 'b', 'c', 'a'], categories=['a', 'b', 'c', 'd'])
    with pytest.raises(ValueError, match=msg):
        ci.reindex(Categorical(['a', 'c']))