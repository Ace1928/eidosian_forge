from statsmodels.compat.pandas import MONTH_END
import warnings
import numpy as np
from numpy.testing import assert_, assert_allclose
import pandas as pd
import pytest
from scipy.stats import ortho_group
from statsmodels.tools.sm_exceptions import EstimationWarning
from statsmodels.tsa.statespace import (
from statsmodels.tsa.vector_ar.tests.test_var import get_macrodata

    Time-varying state space model for testing

    This creates a state space model with randomly generated time-varying
    system matrices. When used in a test, that test should use
    `reset_randomstate` to ensure consistent test runs.
    