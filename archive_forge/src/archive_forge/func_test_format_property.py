import pytest
import numpy as np
import numpy.testing as npt
import scipy.sparse
import scipy.sparse.linalg as spla
from scipy._lib._util import VisibleDeprecationWarning
def test_format_property():
    for fmt in sparray_types:
        arr_cls = getattr(scipy.sparse, f'{fmt}_array')
        M = arr_cls([[1, 2]])
        assert M.format == fmt
        assert M._format == fmt
        with pytest.raises(AttributeError):
            M.format = 'qqq'