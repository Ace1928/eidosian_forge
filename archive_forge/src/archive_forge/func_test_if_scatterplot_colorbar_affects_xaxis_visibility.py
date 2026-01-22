import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_if_scatterplot_colorbar_affects_xaxis_visibility(self):
    random_array = np.random.default_rng(2).random((10, 3))
    df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
    ax1 = df.plot.scatter(x='A label', y='B label')
    ax2 = df.plot.scatter(x='A label', y='B label', c='C label')
    vis1 = [vis.get_visible() for vis in ax1.xaxis.get_minorticklabels()]
    vis2 = [vis.get_visible() for vis in ax2.xaxis.get_minorticklabels()]
    assert vis1 == vis2
    vis1 = [vis.get_visible() for vis in ax1.xaxis.get_majorticklabels()]
    vis2 = [vis.get_visible() for vis in ax2.xaxis.get_majorticklabels()]
    assert vis1 == vis2
    assert ax1.xaxis.get_label().get_visible() == ax2.xaxis.get_label().get_visible()