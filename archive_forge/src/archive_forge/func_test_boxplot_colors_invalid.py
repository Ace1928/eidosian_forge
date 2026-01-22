import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_boxplot_colors_invalid(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    msg = re.escape("color dict contains invalid key 'xxxx'. The key must be either ['boxes', 'whiskers', 'medians', 'caps']")
    with pytest.raises(ValueError, match=msg):
        df.plot.box(color={'boxes': 'red', 'xxxx': 'blue'})