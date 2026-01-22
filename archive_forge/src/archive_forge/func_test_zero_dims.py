import numpy as np
from numpy.testing import (
import pytest
def test_zero_dims(self):
    try:
        np.poly(np.zeros((0, 0)))
    except ValueError:
        pass