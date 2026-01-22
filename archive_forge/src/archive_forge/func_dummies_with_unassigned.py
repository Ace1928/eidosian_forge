import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def dummies_with_unassigned():
    return DataFrame({'col1_a': [1, 0, 0], 'col1_b': [0, 1, 0], 'col2_a': [0, 1, 0], 'col2_b': [0, 0, 0], 'col2_c': [0, 0, 1]})