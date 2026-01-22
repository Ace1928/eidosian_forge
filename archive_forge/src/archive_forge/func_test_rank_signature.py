from itertools import chain
import operator
import numpy as np
import pytest
from pandas._libs.algos import (
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
def test_rank_signature(self):
    s = Series([0, 1])
    s.rank(method='average')
    msg = 'No axis named average for object type Series'
    with pytest.raises(ValueError, match=msg):
        s.rank('average')