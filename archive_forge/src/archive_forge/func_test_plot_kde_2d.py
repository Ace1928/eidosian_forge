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
@pytest.mark.parametrize('kwargs', [{'contour': True, 'fill_last': False}, {'contour': True, 'contourf_kwargs': {'cmap': 'plasma'}}, {'contour': False}, {'contour': False, 'pcolormesh_kwargs': {'cmap': 'plasma'}}, {'contour': True, 'contourf_kwargs': {'levels': 3}}, {'contour': True, 'contourf_kwargs': {'levels': [0.1, 0.2, 0.3]}}, {'hdi_probs': [0.3, 0.9, 0.6]}, {'hdi_probs': [0.3, 0.6, 0.9], 'contourf_kwargs': {'cmap': 'Blues'}}, {'hdi_probs': [0.9, 0.6, 0.3], 'contour_kwargs': {'alpha': 0}}])
def test_plot_kde_2d(continuous_model, kwargs):
    axes = plot_kde(continuous_model['x'], continuous_model['y'], backend='bokeh', show=False, **kwargs)
    assert axes