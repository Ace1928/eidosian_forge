import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_set_ordered(self):
    cat = Categorical(['a', 'b', 'c', 'a'], ordered=True)
    cat2 = cat.as_unordered()
    assert not cat2.ordered
    cat2 = cat.as_ordered()
    assert cat2.ordered
    assert cat2.set_ordered(True).ordered
    assert not cat2.set_ordered(False).ordered
    msg = "property 'ordered' of 'Categorical' object has no setter" if PY311 else "can't set attribute"
    with pytest.raises(AttributeError, match=msg):
        cat.ordered = True
    with pytest.raises(AttributeError, match=msg):
        cat.ordered = False