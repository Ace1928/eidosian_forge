import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
@pytest.fixture(params=['kaiser', 'gaussian', 'general_gaussian', 'exponential'])
def win_types_special(request):
    return request.param