import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_dont_modify_colors(self):
    colors = ['r', 'g', 'b']
    DataFrame(np.random.default_rng(2).random((10, 2))).plot(color=colors)
    assert len(colors) == 3