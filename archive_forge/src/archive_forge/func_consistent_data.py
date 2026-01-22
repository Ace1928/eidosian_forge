import itertools
import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=(obj for obj in itertools.chain(create_series(), create_dataframes()) if is_constant(obj)))
def consistent_data(request):
    return request.param