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
def test_boxplot_vertical(self, hist_df):
    df = hist_df
    numeric_cols = df._get_numeric_data().columns
    labels = [pprint_thing(c) for c in numeric_cols]
    ax = df.plot.box(rot=50, fontsize=8, vert=False)
    _check_ticks_props(ax, xrot=0, yrot=50, ylabelsize=8)
    _check_text_labels(ax.get_yticklabels(), labels)
    assert len(ax.lines) == 7 * len(numeric_cols)