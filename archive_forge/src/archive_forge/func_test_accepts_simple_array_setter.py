import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_accepts_simple_array_setter():

    class Foo(HasTraits):
        bar = NDArray()
    foo = Foo()
    foo.bar = np.zeros((4, 4, 3))
    np.testing.assert_equal(foo.bar, np.zeros((4, 4, 3)))