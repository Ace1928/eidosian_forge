import os
from platform import machine
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import (
from ..testing import suppress_warnings
def test_floor_log2():
    assert floor_log2(2 ** 9 + 1) == 9
    assert floor_log2(-2 ** 9 + 1) == 8
    assert floor_log2(2) == 1
    assert floor_log2(1) == 0
    assert floor_log2(0.5) == -1
    assert floor_log2(0.75) == -1
    assert floor_log2(0.25) == -2
    assert floor_log2(0.24) == -3
    assert floor_log2(0) is None