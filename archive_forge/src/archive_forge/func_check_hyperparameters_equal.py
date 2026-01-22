from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def check_hyperparameters_equal(kernel1, kernel2):
    for attr in set(dir(kernel1) + dir(kernel2)):
        if attr.startswith('hyperparameter_'):
            attr_value1 = getattr(kernel1, attr)
            attr_value2 = getattr(kernel2, attr)
            assert attr_value1 == attr_value2