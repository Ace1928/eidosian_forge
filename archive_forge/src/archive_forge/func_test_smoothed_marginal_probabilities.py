import warnings
import os
import numpy as np
from numpy.testing import assert_equal, assert_allclose
import pandas as pd
import pytest
from statsmodels.tools import add_constant
from statsmodels.tsa.regime_switching import markov_autoregression
def test_smoothed_marginal_probabilities(self):
    assert_allclose(self.result.smoothed_marginal_probabilities[:, 0], self.true['smoothed0'], atol=1e-06)
    assert_allclose(self.result.smoothed_marginal_probabilities[:, 1], self.true['smoothed1'], atol=1e-06)