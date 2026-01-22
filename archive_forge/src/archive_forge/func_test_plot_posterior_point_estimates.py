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
@pytest.mark.parametrize('point_estimate', ('mode', 'mean', 'median'))
def test_plot_posterior_point_estimates(models, point_estimate):
    axes = plot_posterior(models.model_1, var_names=('mu', 'tau'), point_estimate=point_estimate, backend='bokeh', show=False)
    assert axes.shape == (1, 2)