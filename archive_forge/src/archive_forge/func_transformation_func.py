import numpy as np
import pytest
from pandas import (
from pandas.core.groupby.base import (
@pytest.fixture(params=sorted(transformation_kernels))
def transformation_func(request):
    """yields the string names of all groupby transformation functions."""
    return request.param