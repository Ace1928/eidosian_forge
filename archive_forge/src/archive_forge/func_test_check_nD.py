import sys
import warnings
import numpy as np
import pytest
from skimage._shared import testing
from skimage._shared.utils import (
def test_check_nD():
    z = np.random.random(200 ** 2).reshape((200, 200))
    x = z[10:30, 30:10]
    with testing.raises(ValueError):
        check_nD(x, 2)