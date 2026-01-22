import pytest
import numpy as np
from numpy.core._multiarray_tests import argparse_example_function as func
def test_multiple_values():
    with pytest.raises(TypeError, match="given by name \\('arg2'\\) and position \\(position 1\\)"):
        func(1, 2, arg2=3)