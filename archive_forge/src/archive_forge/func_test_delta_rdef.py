import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_delta_rdef(self):
    self.generate('test_delta_Rdef', 'test_lowess_delta.csv', out='out_Rdef', kwargs=lambda data: {'frac': 0.1, 'delta': 0.01 * np.ptp(data['x'])})