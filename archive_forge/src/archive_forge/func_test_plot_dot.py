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
@pytest.mark.parametrize('kwargs', [{}, {'binwidth': 0.5, 'stackratio': 2, 'nquantiles': 20}, {'point_interval': True}, {'point_interval': True, 'dotsize': 1.2, 'point_estimate': 'median', 'plot_kwargs': {'color': 'grey'}}, {'point_interval': True, 'plot_kwargs': {'color': 'grey'}, 'nquantiles': 100, 'hdi_prob': 0.95, 'intervalcolor': 'green'}, {'point_interval': True, 'plot_kwargs': {'color': 'grey'}, 'quartiles': False, 'linewidth': 2}])
def test_plot_dot(continuous_model, kwargs):
    data = continuous_model['x']
    ax = plot_dot(data, **kwargs, backend='bokeh', show=False)
    assert ax