import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_stacked_boxplot_set_axis(self):
    import matplotlib.pyplot as plt
    n = 80
    df = DataFrame({'Clinical': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Confirmed': np.random.default_rng(2).choice([0, 1, 2, 3], n), 'Discarded': np.random.default_rng(2).choice([0, 1, 2, 3], n)}, index=np.arange(0, n))
    ax = df.plot(kind='bar', stacked=True)
    assert [int(x.get_text()) for x in ax.get_xticklabels()] == df.index.to_list()
    ax.set_xticks(np.arange(0, 80, 10))
    plt.draw()
    assert [int(x.get_text()) for x in ax.get_xticklabels()] == list(np.arange(0, 80, 10))