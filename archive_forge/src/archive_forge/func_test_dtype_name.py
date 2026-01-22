import pytest
import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
def test_dtype_name(self):
    assert SF(1.0).name == '_ScaledFloatTestDType64'