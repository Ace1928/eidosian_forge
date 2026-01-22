import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_color_empty_string(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    with pytest.raises(ValueError, match='Invalid color argument:'):
        df.plot(color='')