import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
def test_iter_buffered_cast_structured_type_failure_with_cleanup():
    sdt1 = [('a', 'f4'), ('b', 'i8'), ('d', 'O')]
    sdt2 = [('b', 'O'), ('a', 'f8')]
    a = np.array([(1, 2, 3), (4, 5, 6)], dtype=sdt1)
    for intent in ['readwrite', 'readonly', 'writeonly']:
        simple_arr = np.array([1, 2], dtype='i,i')
        with pytest.raises(TypeError):
            nditer((simple_arr, a), ['buffered', 'refs_ok'], [intent, intent], casting='unsafe', op_dtypes=['f,f', sdt2])