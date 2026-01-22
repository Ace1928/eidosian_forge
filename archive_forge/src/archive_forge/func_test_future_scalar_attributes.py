import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
@pytest.mark.parametrize('name', ['bool', 'long', 'ulong', 'str', 'bytes', 'object'])
def test_future_scalar_attributes(name):
    assert name not in dir(np)
    with pytest.warns(FutureWarning, match=f'In the future .*{name}'):
        assert not hasattr(np, name)
    np.dtype(name)
    name in np.sctypeDict