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
@pytest.mark.parametrize('kwargs', [{}, {'n_unif': 50}, {'use_hdi': True, 'color': 'gray'}, {'use_hdi': True, 'hdi_prob': 0.68}, {'use_hdi': True, 'hdi_kwargs': {'line_dash': 'dashed', 'alpha': 0}}, {'ecdf': True}, {'ecdf': True, 'ecdf_fill': False, 'plot_unif_kwargs': {'line_dash': '--'}}, {'ecdf': True, 'hdi_prob': 0.97, 'fill_kwargs': {'color': 'red'}}])
def test_plot_loo_pit(models, kwargs):
    axes = plot_loo_pit(idata=models.model_1, y='y', backend='bokeh', show=False, **kwargs)
    assert axes