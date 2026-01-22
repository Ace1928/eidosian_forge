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
@pytest.mark.parametrize('args', [{'y': 'str'}, {'y': 'DataArray', 'y_hat': 'str'}, {'y': 'ndarray', 'y_hat': 'str'}, {'y': 'ndarray', 'y_hat': 'DataArray'}, {'y': 'ndarray', 'y_hat': 'ndarray'}])
def test_plot_loo_pit_label(models, args):
    if args['y'] == 'str':
        y = 'y'
    elif args['y'] == 'DataArray':
        y = models.model_1.observed_data.y
    elif args['y'] == 'ndarray':
        y = models.model_1.observed_data.y.values
    if args.get('y_hat') == 'str':
        y_hat = 'y'
    elif args.get('y_hat') == 'DataArray':
        y_hat = models.model_1.posterior_predictive.y.stack(__sample__=('chain', 'draw'))
    elif args.get('y_hat') == 'ndarray':
        y_hat = models.model_1.posterior_predictive.y.stack(__sample__=('chain', 'draw')).values
    else:
        y_hat = None
    ax = plot_loo_pit(idata=models.model_1, y=y, y_hat=y_hat, backend='bokeh', show=False)
    assert ax