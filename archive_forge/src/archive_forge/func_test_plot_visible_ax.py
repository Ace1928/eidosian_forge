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
@pytest.mark.slow
def test_plot_visible_ax(self):
    df = DataFrame(np.random.default_rng(2).random((10, 3)), index=list(string.ascii_letters[:10]))
    axes = df.plot(subplots=True, title='blah')
    _check_axes_shape(axes, axes_num=3, layout=(3, 1))
    for ax in axes[:2]:
        _check_visible(ax.xaxis)
        _check_visible(ax.get_xticklabels(), visible=False)
        _check_visible(ax.get_xticklabels(minor=True), visible=False)
        _check_visible([ax.xaxis.get_label()], visible=False)
    for ax in [axes[2]]:
        _check_visible(ax.xaxis)
        _check_visible(ax.get_xticklabels())
        _check_visible([ax.xaxis.get_label()])
        _check_ticks_props(ax, xrot=0)