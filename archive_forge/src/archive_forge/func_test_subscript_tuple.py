import fractions
import platform
import types
from typing import Any, Type
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_raises, IS_MUSL
@pytest.mark.parametrize('arg_len', range(4))
def test_subscript_tuple(self, arg_len: int) -> None:
    arg_tup = (Any,) * arg_len
    if arg_len == 1:
        assert np.number[arg_tup]
    else:
        with pytest.raises(TypeError):
            np.number[arg_tup]