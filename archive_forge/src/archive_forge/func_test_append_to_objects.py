import pytest
import numpy as np
import numpy.ma as ma
from numpy.ma.mrecords import MaskedRecords
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_, assert_raises
from numpy.lib.recfunctions import (
def test_append_to_objects(self):
    """Test append_fields when the base array contains objects"""
    obj = self.data['obj']
    x = np.array([(obj, 1.0), (obj, 2.0)], dtype=[('A', object), ('B', float)])
    y = np.array([10, 20], dtype=int)
    test = append_fields(x, 'C', data=y, usemask=False)
    control = np.array([(obj, 1.0, 10), (obj, 2.0, 20)], dtype=[('A', object), ('B', float), ('C', int)])
    assert_equal(test, control)