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
@pytest.mark.parametrize('kwargs', [{}, {'var_names': 'mu'}, {'var_names': ('mu', 'tau')}, {'rope': (-2, 2)}, {'rope': {'mu': [{'rope': (-2, 2)}], 'theta': [{'school': 'Choate', 'rope': (2, 4)}]}}, {'point_estimate': 'mode'}, {'point_estimate': 'median'}, {'point_estimate': None}, {'hdi_prob': 'hide', 'legend_label': ''}, {'ref_val': 0}, {'ref_val': None}, {'ref_val': {'mu': [{'ref_val': 1}]}}, {'bins': None, 'kind': 'hist'}, {'ref_val': {'theta': [{'school': 'Lawrenceville', 'ref_val': 3}]}}])
def test_plot_posterior(models, kwargs):
    axes = plot_posterior(models.model_1, backend='bokeh', show=False, **kwargs)
    assert axes.shape