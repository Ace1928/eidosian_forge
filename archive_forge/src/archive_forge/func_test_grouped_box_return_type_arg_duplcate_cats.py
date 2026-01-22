import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('return_type', ['dict', 'axes', 'both'])
def test_grouped_box_return_type_arg_duplcate_cats(self, return_type):
    columns2 = 'X B C D A'.split()
    df2 = DataFrame(np.random.default_rng(2).standard_normal((6, 5)), columns=columns2)
    categories2 = 'A B'.split()
    df2['category'] = categories2 * 3
    returned = df2.groupby('category').boxplot(return_type=return_type)
    _check_box_return_type(returned, return_type, expected_keys=categories2)
    returned = df2.boxplot(by='category', return_type=return_type)
    _check_box_return_type(returned, return_type, expected_keys=columns2)