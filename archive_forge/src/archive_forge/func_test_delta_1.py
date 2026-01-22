import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_delta_1(self):
    self.generate('test_delta_1', 'test_lowess_delta.csv', out='out_1', kwargs={'frac': 0.1, 'delta': 1 + 1e-10}, decimal=10)