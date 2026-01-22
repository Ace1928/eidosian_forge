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
def test_pie_df_labels_colors(self):
    df = DataFrame(np.random.default_rng(2).random((5, 3)), columns=['X', 'Y', 'Z'], index=['a', 'b', 'c', 'd', 'e'])
    labels = ['A', 'B', 'C', 'D', 'E']
    color_args = ['r', 'g', 'b', 'c', 'm']
    axes = _check_plot_works(df.plot.pie, default_axes=True, subplots=True, labels=labels, colors=color_args)
    assert len(axes) == len(df.columns)
    for ax in axes:
        _check_text_labels(ax.texts, labels)
        _check_colors(ax.patches, facecolors=color_args)