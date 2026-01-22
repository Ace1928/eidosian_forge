from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_fix_unfixable(ses):
    mod = ExponentialSmoothing(ses, initialization_method='estimated')
    with pytest.raises(ValueError, match='Cannot fix a parameter'):
        with mod.fix_params({'smoothing_level': 0.25}):
            mod.fit(smoothing_level=0.2)