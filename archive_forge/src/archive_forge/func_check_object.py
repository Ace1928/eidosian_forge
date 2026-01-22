from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
import os
from statsmodels import datasets
from statsmodels.tsa.statespace.initialization import Initialization
from statsmodels.tsa.statespace.kalman_smoother import KalmanSmoother
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from numpy.testing import assert_equal, assert_allclose
from . import kfas_helpers
def check_object(self, actual, desired, rtol_diffuse):
    if actual is None or desired is None:
        return
    d = None
    if rtol_diffuse is None:
        rtol_diffuse = self.rtol_diffuse
    if rtol_diffuse is not None:
        d = self.d
        if rtol_diffuse != np.inf:
            assert_allclose(actual.T[:d], desired.T[:d], rtol=rtol_diffuse, atol=self.atol_diffuse)
    assert_allclose(actual.T[d:], desired.T[d:], rtol=self.rtol, atol=self.atol)