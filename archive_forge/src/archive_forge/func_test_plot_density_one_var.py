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
def test_plot_density_one_var():
    """Test plot_density works when there is only one variable (#1401)."""
    model_ab = from_dict({'a': np.random.normal(size=200)})
    model_bc = from_dict({'a': np.random.normal(size=200)})
    axes = plot_density([model_ab, model_bc], backend='bokeh', show=False)
    assert axes.size == 1