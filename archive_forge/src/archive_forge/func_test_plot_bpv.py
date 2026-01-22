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
@pytest.mark.parametrize('kwargs', [{}, {'reference': 'analytical'}, {'kind': 'p_value'}, {'kind': 't_stat', 't_stat': 'std'}, {'kind': 't_stat', 't_stat': 0.5, 'bpv': True}])
def test_plot_bpv(models, kwargs):
    axes = plot_bpv(models.model_1, backend='bokeh', show=False, **kwargs)
    assert axes.shape