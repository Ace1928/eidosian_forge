import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_if_hexbin_xaxis_label_is_visible(self):
    random_array = np.random.default_rng(2).random((10, 3))
    df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
    ax = df.plot.hexbin('A label', 'B label', gridsize=12)
    assert all((vis.get_visible() for vis in ax.xaxis.get_minorticklabels()))
    assert all((vis.get_visible() for vis in ax.xaxis.get_majorticklabels()))
    assert ax.xaxis.get_label().get_visible()