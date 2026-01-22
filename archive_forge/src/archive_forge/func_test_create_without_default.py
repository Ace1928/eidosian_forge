import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_create_without_default():

    class Foo(HasTraits):
        bar = NDArray()
    foo = Foo()
    assert foo.bar is Undefined
    foo.bar = np.zeros((4, 4))
    np.testing.assert_equal(foo.bar, np.zeros((4, 4)))