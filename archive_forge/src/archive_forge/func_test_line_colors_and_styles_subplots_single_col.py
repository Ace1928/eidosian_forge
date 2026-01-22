import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_line_colors_and_styles_subplots_single_col(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.loc[:, [0]].plot(color='DodgerBlue', subplots=True)
    _check_colors(axes[0].lines, linecolors=['DodgerBlue'])