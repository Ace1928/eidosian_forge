import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..casting import sctypes, type_info
from ..testing import suppress_warnings
from ..volumeutils import apply_read_scaling, array_from_file, array_to_file, finite_range
from .test_volumeutils import _calculate_scale
def test_finite_range_err():
    a = np.array([[1.0, 0, 1], [2, 3, 4]]).view([('f1', 'f')])
    with pytest.raises(TypeError):
        finite_range(a)