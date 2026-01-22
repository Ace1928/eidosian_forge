import os
import numpy as np
from numpy.testing import (
import pytest
from statsmodels.nonparametric.smoothers_lowess import lowess
import pandas as pd
def test_iter_0(self):
    self.generate('test_iter_0', 'test_lowess_iter.csv', out='out_0', kwargs={'it': 0})