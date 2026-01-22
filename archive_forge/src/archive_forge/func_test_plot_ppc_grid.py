from copy import deepcopy
import numpy as np
import pytest
from pandas import DataFrame  # pylint: disable=wrong-import-position
from scipy.stats import norm  # pylint: disable=wrong-import-position
from ...data import from_dict, load_arviz_data  # pylint: disable=wrong-import-position
from ...plots import (  # pylint: disable=wrong-import-position
from ...rcparams import rc_context, rcParams  # pylint: disable=wrong-import-position
from ...stats import compare, hdi, loo, waic  # pylint: disable=wrong-import-position
from ..helpers import (  # pylint: disable=unused-import, wrong-import-position
def test_plot_ppc_grid(models):
    axes = plot_ppc(models.model_1, kind='scatter', flatten=[], backend='bokeh', show=False)
    assert len(axes.ravel()) == 8
    axes = plot_ppc(models.model_1, kind='scatter', flatten=[], coords={'obs_dim': [1, 2, 3]}, backend='bokeh', show=False)
    assert len(axes.ravel()) == 3
    axes = plot_ppc(models.model_1, kind='scatter', flatten=['obs_dim'], coords={'obs_dim': [1, 2, 3]}, backend='bokeh', show=False)
    assert len(axes.ravel()) == 1