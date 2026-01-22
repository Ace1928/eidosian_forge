import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=['D', '3D', '-3D', 'h', '2h', '-2h', 'min', '2min', 's', '-3s'])
def freq_sample(request):
    """
    Valid values for 'freq' parameter used to create date_range and
    timedelta_range..
    """
    return request.param