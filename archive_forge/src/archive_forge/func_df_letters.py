from string import ascii_lowercase
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.groupby.base import (
@pytest.fixture
def df_letters():
    letters = np.array(list(ascii_lowercase))
    N = 10
    random_letters = letters.take(np.random.randint(0, 26, N))
    df = DataFrame({'floats': N / 10 * Series(np.random.random(N)), 'letters': Series(random_letters)})
    return df