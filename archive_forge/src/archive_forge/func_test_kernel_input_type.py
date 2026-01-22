from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
@pytest.mark.parametrize('kernel', kernels)
def test_kernel_input_type(kernel):
    if isinstance(kernel, Exponentiation):
        assert kernel.requires_vector_input == kernel.kernel.requires_vector_input
    if isinstance(kernel, KernelOperator):
        assert kernel.requires_vector_input == (kernel.k1.requires_vector_input or kernel.k2.requires_vector_input)