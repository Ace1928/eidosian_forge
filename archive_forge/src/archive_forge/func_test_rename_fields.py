import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_rename_fields(self):
    a = np.array([(1, (2, [3.0, 30.0])), (4, (5, [6.0, 60.0]))], dtype=[('a', int), ('b', [('ba', float), ('bb', (float, 2))])])
    test = rename_fields(a, {'a': 'A', 'bb': 'BB'})
    newdtype = [('A', int), ('b', [('ba', float), ('BB', (float, 2))])]
    control = a.view(newdtype)
    assert_equal(test.dtype, newdtype)
    assert_equal(test, control)