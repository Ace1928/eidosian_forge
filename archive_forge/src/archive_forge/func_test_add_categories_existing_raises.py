import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_add_categories_existing_raises(self):
    cat = Categorical(['a', 'b', 'c', 'd'], ordered=True)
    msg = re.escape("new categories must not include old categories: {'d'}")
    with pytest.raises(ValueError, match=msg):
        cat.add_categories(['d'])