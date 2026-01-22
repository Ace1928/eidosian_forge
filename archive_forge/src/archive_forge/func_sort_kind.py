import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=['quicksort', 'mergesort', 'heapsort', 'stable'])
def sort_kind(request):
    return request.param