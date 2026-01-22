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
@pytest.mark.parametrize('kwargs', [{}, {'var_names': 'mu'}, {'var_names': ('mu', 'tau'), 'coords': {'school': [0, 1]}}, {'var_names': 'mu', 'ref_line': True}, {'var_names': 'mu', 'ref_line_kwargs': {'line_width': 2, 'line_color': 'red'}, 'bar_kwargs': {'width': 50}}, {'var_names': 'mu', 'ref_line': False}, {'var_names': 'mu', 'kind': 'vlines'}, {'var_names': 'mu', 'kind': 'vlines', 'vlines_kwargs': {'line_width': 0}, 'marker_vlines_kwargs': {'radius': 20}}])
def test_plot_rank(models, kwargs):
    axes = plot_rank(models.model_1, backend='bokeh', show=False, **kwargs)
    assert axes.shape