import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_class_discovery(self):
    dt, _ = discover_array_params([1.0, 2.0, 3.0], dtype=SF)
    assert dt == SF(1.0)