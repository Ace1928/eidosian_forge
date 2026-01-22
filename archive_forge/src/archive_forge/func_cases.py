import operator
import numpy as np
import pytest
import pandas._libs.sparse as splib
import pandas.util._test_decorators as td
from pandas import Series
import pandas._testing as tm
from pandas.core.arrays.sparse import (
@pytest.fixture(params=[[[0, 7, 15], [3, 5, 5], [2, 9, 14], [2, 3, 5], [2, 9, 15], [1, 3, 4]], [[0, 5], [4, 4], [1], [4], [1], [3]], [[0], [10], [0, 5], [3, 7], [0, 5], [3, 5]], [[10], [5], [0, 12], [5, 3], [12], [3]], [[0, 10], [4, 6], [5, 17], [4, 2], [], []], [[0], [5], [], [], [], []]], ids=['plain_case', 'delete_blocks', 'split_blocks', 'skip_block', 'no_intersect', 'one_empty'])
def cases(request):
    return request.param