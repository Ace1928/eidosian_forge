import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
@pytest.mark.matplotlib
def test_plot_coefficients_of_determination(self, close_figures):
    self.results.plot_coefficients_of_determination()