import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_return_type_legacy(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((6, 4)), index=list(string.ascii_letters[:6]), columns=['one', 'two', 'three', 'four'])
    msg = "return_type must be {'axes', 'dict', 'both'}"
    with pytest.raises(ValueError, match=msg):
        df.boxplot(return_type='NOT_A_TYPE')
    result = df.boxplot()
    _check_box_return_type(result, 'axes')