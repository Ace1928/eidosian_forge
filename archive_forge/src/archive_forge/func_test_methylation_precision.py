import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_methylation_precision(self):
    rslt = self.meth_log_fit
    assert_allclose(rslt.params[-2:], expected_methylation_precision['Estimate'], atol=1e-05, rtol=1e-10)