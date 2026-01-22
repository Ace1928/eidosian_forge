import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
@pytest.mark.parametrize('axis', [None, 0])
def test_subarray_error(self, axis):
    with pytest.raises(TypeError, match='.*subarray dtype'):
        np.concatenate(([1], [1]), dtype='(2,)i', axis=axis)