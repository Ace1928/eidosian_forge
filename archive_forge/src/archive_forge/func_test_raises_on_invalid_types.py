from typing import Callable
import pytest
from numpy import array_api as xp
@pytest.mark.parametrize('func, args, kwargs', [p(xp.can_cast, 42, xp.int8), p(xp.can_cast, xp.int8, 42), p(xp.result_type, 42)])
def test_raises_on_invalid_types(func, args, kwargs):
    """Function raises TypeError when passed invalidly-typed inputs"""
    with pytest.raises(TypeError):
        func(*args, **kwargs)