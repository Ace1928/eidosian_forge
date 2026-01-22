import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=[list, tuple, np.array, array, Series])
def listlike_box(request):
    """
    Types that may be passed as the indexer to searchsorted.
    """
    return request.param