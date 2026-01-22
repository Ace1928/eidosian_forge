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
@pytest.mark.parametrize('kwargs', [{'color': 'C5', 'circular': True}, {'hdi_data': True, 'fill_kwargs': {'alpha': 0}}, {'plot_kwargs': {'alpha': 0}}, {'smooth_kwargs': {'window_length': 33, 'polyorder': 5, 'mode': 'mirror'}}, {'hdi_data': True, 'smooth': False, 'color': 'xkcd:jade'}])
def test_plot_hdi(models, data, kwargs):
    hdi_data = kwargs.pop('hdi_data', None)
    y_data = models.model_1.posterior['theta']
    if hdi_data:
        hdi_data = hdi(y_data)
        axis = plot_hdi(data['y'], hdi_data=hdi_data, backend='bokeh', show=False, **kwargs)
    else:
        axis = plot_hdi(data['y'], y_data, backend='bokeh', show=False, **kwargs)
    assert axis