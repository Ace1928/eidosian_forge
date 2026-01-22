from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.parametrize('markers', [{0: '^', 1: '+', 2: 'o'}, {0: '^', 1: '+'}, ['^', '+', 'o'], ['^', '+']])
def test_style_by_column(self, markers):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    fig.clf()
    fig.add_subplot(111)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)))
    ax = df.plot(style=markers)
    for idx, line in enumerate(ax.get_lines()[:len(markers)]):
        assert line.get_marker() == markers[idx]