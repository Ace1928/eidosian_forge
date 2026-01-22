import operator
import pytest
from pandas._config.config import _get_option
from pandas import (
@pytest.fixture(params=[lambda x: 1, lambda x: [1] * len(x), lambda x: Series([1] * len(x)), lambda x: x], ids=['scalar', 'list', 'series', 'object'])
def groupby_apply_op(request):
    """
    Functions to test groupby.apply().
    """
    return request.param