import sys
import gc
from hypothesis import given
from hypothesis.extra import numpy as hynp
import pytest
import numpy as np
from numpy.testing import (
from numpy.core.arrayprint import _typelessdata
import textwrap
def test_fieldless_structured(self):
    no_fields = np.dtype([])
    arr_no_fields = np.empty(4, dtype=no_fields)
    assert_equal(repr(arr_no_fields), 'array([(), (), (), ()], dtype=[])')