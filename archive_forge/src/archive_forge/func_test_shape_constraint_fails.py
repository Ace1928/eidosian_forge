import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
@pytest.mark.parametrize('constraints', [(4, None, None), (None, 2, None), (None, None, 3), (4, 2, None), (4, None, 3), (None, 2, 3), (4, 2, 3), (4, 2, 3, 33, 432), (1, 5, 3, 2), (5, 3, 2, 1)])
def test_shape_constraint_fails(constraints):

    class Foo(HasTraits):
        bar = NDArray().valid(shape_constraints(*constraints))
    foo = Foo()
    assert foo.bar is Undefined
    with pytest.raises(TraitError):
        foo.bar = np.zeros((5, 3, 2))