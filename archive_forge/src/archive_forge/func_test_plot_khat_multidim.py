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
@pytest.mark.parametrize('kwargs', [{}, {'xlabels': True}, {'color': 'dim1', 'xlabels': True, 'show_bins': True, 'bin_format': '{0}'}, {'color': 'dim2', 'legend': True, 'hover_label': True}, {'color': 'blue', 'coords': {'dim2': slice(2, 4)}}, {'color': np.random.uniform(size=35), 'show_bins': True}, {'color': np.random.uniform(size=(35, 3)), 'show_bins': True, 'show_hlines': True, 'threshold': 1}])
@pytest.mark.parametrize('input_type', ['elpd_data', 'data_array', 'array'])
def test_plot_khat_multidim(multidim_models, input_type, kwargs):
    khats_data = loo(multidim_models.model_1, pointwise=True)
    if input_type == 'data_array':
        khats_data = khats_data.pareto_k
    elif input_type == 'array':
        khats_data = khats_data.pareto_k.values
        if 'color' in kwargs and isinstance(kwargs['color'], str) and (kwargs['color'] in ('dim1', 'dim2')):
            kwargs['color'] = None
    axes = plot_khat(khats_data, backend='bokeh', show=False, **kwargs)
    assert axes