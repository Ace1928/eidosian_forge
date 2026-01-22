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
def test_plot_loo_pit_incompatible_args(models):
    """Test error when both ecdf and use_hdi are True."""
    with pytest.raises(ValueError, match='incompatible'):
        plot_loo_pit(idata=models.model_1, y='y', ecdf=True, use_hdi=True, backend='bokeh', show=False)