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
@pytest.mark.parametrize('kwargs', [{'point_estimate': 'mean'}, {'point_estimate': 'median'}, {'hdi_prob': 0.94}, {'hdi_prob': 1}, {'outline': True}, {'hdi_markers': ['v']}, {'shade': 1}])
def test_plot_density_float(models, kwargs):
    obj = [getattr(models, model_fit) for model_fit in ['model_1', 'model_2']]
    axes = plot_density(obj, backend='bokeh', show=False, **kwargs)
    assert axes.shape[0] >= 6
    assert axes.shape[0] >= 3