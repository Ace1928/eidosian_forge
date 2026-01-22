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
@pytest.mark.parametrize('kwargs', [{}, {'y_hat': 'bad_name'}, {'x': 'x1'}, {'x': ('x1', 'x2')}, {'x': ('x1', 'x2'), 'y_kwargs': {'fill_color': 'blue'}, 'y_hat_plot_kwargs': {'fill_color': 'orange'}, 'legend': True}, {'x': ('x1', 'x2'), 'y_model_plot_kwargs': {'line_color': 'red'}}, {'x': ('x1', 'x2'), 'kind_pp': 'hdi', 'kind_model': 'hdi', 'y_model_fill_kwargs': {'color': 'red'}, 'y_hat_fill_kwargs': {'color': 'cyan'}}])
def test_plot_lm(models, kwargs):
    """Test functionality for 1D data."""
    idata = models.model_1
    if 'constant_data' not in idata.groups():
        y = idata.observed_data['y']
        x1data = y.coords[y.dims[0]]
        idata.add_groups({'constant_data': {'_': x1data}})
        idata.constant_data['x1'] = x1data
        idata.constant_data['x2'] = x1data
    axes = plot_lm(idata=idata, y='y', y_model='eta', backend='bokeh', xjitter=True, show=False, **kwargs)
    assert np.all(axes)