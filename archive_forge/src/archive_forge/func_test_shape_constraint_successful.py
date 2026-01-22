import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
@pytest.mark.parametrize('constraints', [(4, None, None), (None, 2, None), (None, None, 3), (4, 2, None), (4, None, 3), (None, 2, 3), (4, 2, 3)])
def test_shape_constraint_successful(constraints):

    class Foo(HasTraits):
        bar = NDArray(allow_none=True).valid(shape_constraints(*constraints))
    foo = Foo()
    foo.bar = np.zeros((4, 2, 3))
    foo.bar = None