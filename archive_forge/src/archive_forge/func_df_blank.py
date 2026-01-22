import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.fixture
def df_blank():
    return DataFrame([[0, 0], [0, 0]], columns=['A', 'B'], index=['X', 'Y'])