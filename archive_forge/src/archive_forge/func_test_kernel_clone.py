from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_clone(kernel):
    kernel_cloned = clone(kernel)
    assert kernel == kernel_cloned
    assert id(kernel) != id(kernel_cloned)
    assert kernel.get_params() == kernel_cloned.get_params()
    check_hyperparameters_equal(kernel, kernel_cloned)