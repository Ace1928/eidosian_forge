from datetime import datetime
import numpy as np
import pytest
from pandas import (
@pytest.fixture(params=downsample_methods)
def downsample_method(request):
    """Fixture for parametrization of Grouper downsample methods."""
    return request.param