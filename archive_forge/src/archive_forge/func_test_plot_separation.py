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
@pytest.mark.parametrize('kwargs', [{}, {'y_hat_line': True}, {'expected_events': True}, {'y_hat_line_kwargs': {'linestyle': 'dotted'}}, {'exp_events_kwargs': {'marker': 'o'}}])
def test_plot_separation(kwargs):
    idata = load_arviz_data('classification10d')
    ax = plot_separation(idata=idata, y='outcome', backend='bokeh', show=False, **kwargs)
    assert ax