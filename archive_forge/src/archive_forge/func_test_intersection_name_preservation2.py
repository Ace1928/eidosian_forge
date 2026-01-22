from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
@pytest.mark.parametrize('index', ['string'], indirect=True)
@pytest.mark.parametrize('first_name,second_name,expected_name', [('A', 'A', 'A'), ('A', 'B', None), (None, 'B', None)])
def test_intersection_name_preservation2(self, index, first_name, second_name, expected_name, sort):
    first = index[5:20]
    second = index[:10]
    first.name = first_name
    second.name = second_name
    intersect = first.intersection(second, sort=sort)
    assert intersect.name == expected_name