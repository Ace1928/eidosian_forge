import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_field_assignment_by_name(self):
    a = np.ones(2, dtype=[('a', 'i4'), ('b', 'f8'), ('c', 'u1')])
    newdt = [('b', 'f4'), ('c', 'u1')]
    assert_equal(require_fields(a, newdt), np.ones(2, newdt))
    b = np.array([(1, 2), (3, 4)], dtype=newdt)
    assign_fields_by_name(a, b, zero_unassigned=False)
    assert_equal(a, np.array([(1, 1, 2), (1, 3, 4)], dtype=a.dtype))
    assign_fields_by_name(a, b)
    assert_equal(a, np.array([(0, 1, 2), (0, 3, 4)], dtype=a.dtype))
    a = np.ones(2, dtype=[('a', [('b', 'f8'), ('c', 'u1')])])
    newdt = [('a', [('c', 'u1')])]
    assert_equal(require_fields(a, newdt), np.ones(2, newdt))
    b = np.array([((2,),), ((3,),)], dtype=newdt)
    assign_fields_by_name(a, b, zero_unassigned=False)
    assert_equal(a, np.array([((1, 2),), ((1, 3),)], dtype=a.dtype))
    assign_fields_by_name(a, b)
    assert_equal(a, np.array([((0, 2),), ((0, 3),)], dtype=a.dtype))
    a, b = (np.array(3), np.array(0))
    assign_fields_by_name(b, a)
    assert_equal(b[()], 3)