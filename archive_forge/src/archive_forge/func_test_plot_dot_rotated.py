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
@pytest.mark.parametrize('kwargs', [{'rotated': True}, {'point_interval': True, 'rotated': True, 'dotcolor': 'grey', 'binwidth': 0.5}, {'rotated': True, 'point_interval': True, 'plot_kwargs': {'color': 'grey'}, 'nquantiles': 100, 'dotsize': 0.8, 'hdi_prob': 0.95, 'intervalcolor': 'green'}])
def test_plot_dot_rotated(continuous_model, kwargs):
    data = continuous_model['x']
    ax = plot_dot(data, **kwargs, backend='bokeh', show=False)
    assert ax