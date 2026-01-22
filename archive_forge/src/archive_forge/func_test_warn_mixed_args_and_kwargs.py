import collections
import platform
from unittest import mock
import warnings
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.legend as mlegend
from matplotlib import _api, rc_context
from matplotlib.font_manager import FontProperties
def test_warn_mixed_args_and_kwargs(self):
    fig, ax = plt.subplots()
    th = np.linspace(0, 2 * np.pi, 1024)
    lns, = ax.plot(th, np.sin(th), label='sin')
    lnc, = ax.plot(th, np.cos(th), label='cos')
    with pytest.warns(UserWarning) as record:
        ax.legend((lnc, lns), labels=('a', 'b'))
    assert len(record) == 1
    assert str(record[0].message) == 'You have mixed positional and keyword arguments, some input may be discarded.'