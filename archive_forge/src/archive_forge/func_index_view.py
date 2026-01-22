import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.copy_view.util import get_array
def index_view(index_data=[1, 2]):
    df = DataFrame({'a': index_data, 'b': 1.5})
    view = df[:]
    df = df.set_index('a', drop=True)
    idx = df.index
    return (idx, view)