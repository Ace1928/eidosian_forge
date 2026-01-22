import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=[True])
def nopython(request):
    """nopython keyword argument for numba.jit"""
    return request.param