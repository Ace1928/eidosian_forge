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
@pytest.mark.parametrize('data', [{'A': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([10, 9, 8, 7, 6])), 'B': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([8, 8, 8, 8, 8])), 'C': np.repeat(np.array([1, 2, 3, 4, 5]), np.array([6, 7, 8, 9, 10]))}, {'A': np.repeat(np.array([np.nan, 1, 2, 3, 4, 5]), np.array([3, 10, 9, 8, 7, 6])), 'B': np.repeat(np.array([1, np.nan, 2, 3, 4, 5]), np.array([8, 3, 8, 8, 8, 8])), 'C': np.repeat(np.array([1, 2, 3, np.nan, 4, 5]), np.array([6, 7, 8, 3, 9, 10]))}])
def test_hist_df_coord(self, data):
    df = DataFrame(data)
    ax = df.plot.hist(bins=5)
    self._check_box_coord(ax.patches[:5], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(ax.patches[5:10], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(ax.patches[10:], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([6, 7, 8, 9, 10]))
    ax = df.plot.hist(bins=5, stacked=True)
    self._check_box_coord(ax.patches[:5], expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(ax.patches[5:10], expected_y=np.array([10, 9, 8, 7, 6]), expected_h=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(ax.patches[10:], expected_y=np.array([18, 17, 16, 15, 14]), expected_h=np.array([6, 7, 8, 9, 10]))
    axes = df.plot.hist(bins=5, stacked=True, subplots=True)
    self._check_box_coord(axes[0].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(axes[1].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(axes[2].patches, expected_y=np.array([0, 0, 0, 0, 0]), expected_h=np.array([6, 7, 8, 9, 10]))
    ax = df.plot.hist(bins=5, orientation='horizontal')
    self._check_box_coord(ax.patches[:5], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(ax.patches[5:10], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(ax.patches[10:], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([6, 7, 8, 9, 10]))
    ax = df.plot.hist(bins=5, stacked=True, orientation='horizontal')
    self._check_box_coord(ax.patches[:5], expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(ax.patches[5:10], expected_x=np.array([10, 9, 8, 7, 6]), expected_w=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(ax.patches[10:], expected_x=np.array([18, 17, 16, 15, 14]), expected_w=np.array([6, 7, 8, 9, 10]))
    axes = df.plot.hist(bins=5, stacked=True, subplots=True, orientation='horizontal')
    self._check_box_coord(axes[0].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([10, 9, 8, 7, 6]))
    self._check_box_coord(axes[1].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([8, 8, 8, 8, 8]))
    self._check_box_coord(axes[2].patches, expected_x=np.array([0, 0, 0, 0, 0]), expected_w=np.array([6, 7, 8, 9, 10]))