import re
import sys
import numpy as np
import pytest
from pandas.compat import PYPY
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
@pytest.mark.parametrize('kwarg', ['axis', 'out', 'keepdims'])
@pytest.mark.parametrize('method', ['min', 'max'])
def test_numpy_min_max_unsupported_kwargs_raises(self, method, kwarg):
    cat = Categorical(['a', 'b', 'c', 'b'], ordered=True)
    msg = f"the '{kwarg}' parameter is not supported in the pandas implementation of {method}"
    if kwarg == 'axis':
        msg = '`axis` must be fewer than the number of dimensions \\(1\\)'
    kwargs = {kwarg: 42}
    method = getattr(np, method)
    with pytest.raises(ValueError, match=msg):
        method(cat, **kwargs)