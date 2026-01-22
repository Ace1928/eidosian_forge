import sys
import numpy as np
import six
from six.moves import cStringIO as StringIO
from .compat import optional_dep_ok
def test_wide_dtype_for_and_widen():
    assert np.allclose(widen([1, 2, 3]), [1, 2, 3])
    assert widen([1, 2, 3]).dtype == widest_float
    assert np.allclose(widen([1.0, 2.0, 3.0]), [1, 2, 3])
    assert widen([1.0, 2.0, 3.0]).dtype == widest_float
    assert np.allclose(widen([1 + 0j, 2, 3]), [1, 2, 3])
    assert widen([1 + 0j, 2, 3]).dtype == widest_complex
    import pytest
    pytest.raises(ValueError, widen, ['hi'])