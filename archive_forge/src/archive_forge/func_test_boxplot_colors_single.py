import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_boxplot_colors_single(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    bp = df.plot.box(color='DodgerBlue', return_type='dict')
    _check_colors_box(bp, 'DodgerBlue', 'DodgerBlue', 'DodgerBlue', 'DodgerBlue')