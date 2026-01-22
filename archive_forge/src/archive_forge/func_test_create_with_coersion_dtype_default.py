import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_create_with_coersion_dtype_default():

    class Foo(HasTraits):
        bar = NDArray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    foo = Foo()
    np.testing.assert_equal(foo.bar, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32))