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
@pytest.mark.slow
@pytest.mark.parametrize('kwargs', [{'var_names': 'theta', 'divergences': True, 'coords': {'school': [0, 1]}}, {'divergences': True, 'var_names': ['theta', 'mu']}, {'kind': 'kde', 'var_names': ['theta']}, {'kind': 'hexbin', 'var_names': ['theta']}, {'kind': 'hexbin', 'var_names': ['theta']}, {'kind': 'hexbin', 'var_names': ['theta'], 'coords': {'school': [0, 1]}, 'textsize': 20}, {'point_estimate': 'mean', 'reference_values': {'mu': 0, 'tau': 0}, 'reference_values_kwargs': {'line_color': 'blue'}}])
def test_plot_pair(models, kwargs):
    ax = plot_pair(models.model_1, backend='bokeh', show=False, **kwargs)
    assert np.any(ax)