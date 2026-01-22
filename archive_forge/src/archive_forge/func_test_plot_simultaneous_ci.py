from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
@pytest.mark.smoke
@pytest.mark.matplotlib
def test_plot_simultaneous_ci(self, close_figures):
    self.res._simultaneous_ci()
    reference = self.res.groupsunique[1]
    self.res.plot_simultaneous(comparison_name=reference)