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
@pytest.mark.parametrize('kind', ('line', 'bar', 'barh', 'hist', 'kde', 'density', 'area', 'pie'))
def test_group_subplot(self, kind):
    pytest.importorskip('scipy')
    d = {'a': np.arange(10), 'b': np.arange(10) + 1, 'c': np.arange(10) + 1, 'd': np.arange(10), 'e': np.arange(10)}
    df = DataFrame(d)
    axes = df.plot(subplots=[('b', 'e'), ('c', 'd')], kind=kind)
    assert len(axes) == 3
    expected_labels = (['b', 'e'], ['c', 'd'], ['a'])
    for ax, labels in zip(axes, expected_labels):
        if kind != 'pie':
            _check_legend_labels(ax, labels=labels)
        if kind == 'line':
            assert len(ax.lines) == len(labels)