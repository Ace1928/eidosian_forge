import os
import pytest
from nipype.testing import example_data
from nipype.algorithms.confounds import FramewiseDisplacement, ComputeDVARS, is_outlier
import numpy as np
def test_outliers():
    np.random.seed(0)
    in_data = np.random.randn(100)
    in_data[0] += 10
    assert is_outlier(in_data) == 1