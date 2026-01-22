import re
import textwrap
import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_equal
from skimage.transform import (
from skimage.transform._geometric import (
def test_polynomial_inverse():
    with pytest.raises(NotImplementedError):
        PolynomialTransform().inverse(0)