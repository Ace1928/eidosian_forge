import itertools
import sys
import numpy as np
import pytest
import opt_einsum as oe
def test_bad_path_option():
    with pytest.raises(KeyError):
        oe.contract('a,b,c', [1], [2], [3], optimize='optimall')