import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_initial_states(self):
    mask = results_states.columns.str.startswith(self.name)
    desired = results_states.loc[:, mask].dropna().iloc[0]
    assert_allclose(self.res.initial_state.iloc[0], desired)