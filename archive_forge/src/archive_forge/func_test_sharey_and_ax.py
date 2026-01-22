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
def test_sharey_and_ax(self):
    import matplotlib.pyplot as plt
    gs, axes = _generate_4_axes_via_gridspec()
    df = DataFrame({'a': [1, 2, 3, 4, 5, 6], 'b': [1, 2, 3, 4, 5, 6], 'c': [1, 2, 3, 4, 5, 6], 'd': [1, 2, 3, 4, 5, 6]})

    def _check(axes):
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)
        for ax in [axes[0], axes[1]]:
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[2], axes[3]]:
            _check_visible(ax.get_yticklabels(), visible=False)
    for ax in axes:
        df.plot(x='a', y='b', title='title', ax=ax, sharey=True)
    gs.tight_layout(plt.gcf())
    _check(axes)
    plt.close('all')
    gs, axes = _generate_4_axes_via_gridspec()
    with tm.assert_produces_warning(UserWarning):
        axes = df.plot(subplots=True, ax=axes, sharey=True)
    gs.tight_layout(plt.gcf())
    _check(axes)