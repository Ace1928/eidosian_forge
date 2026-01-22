import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain
import numpy as np
from numpy.testing import (
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle
@pytest.mark.xfail(IS_WASM, reason='not sure why')
@pytest.mark.parametrize('index', [np.ones(10, dtype=bool), np.arange(10)], ids=['boolean-arr-index', 'integer-arr-index'])
def test_nonarray_assignment(self, index):
    a = np.arange(10)
    with pytest.raises(ValueError):
        a[index] = np.nan
    with np.errstate(invalid='warn'):
        with pytest.warns(RuntimeWarning, match='invalid value'):
            a[index] = np.array(np.nan)