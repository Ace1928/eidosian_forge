import numpy as np
import pytest
from pandas import (
@pytest.fixture
def frame_empty():
    return DataFrame()