import re
import numpy as np
import pytest
from pandas.compat import PY311
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.categorical import recode_for_categories
def test_to_list_deprecated(self):
    cat1 = Categorical(list('acb'), ordered=False)
    msg = 'Categorical.to_list is deprecated and will be removed'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        cat1.to_list()