import re
import pytest
import numpy as np
from traitlets import HasTraits, TraitError, Undefined
from ..ndarray.traits import NDArray, shape_constraints
def test_allowed_none():

    class Foo(HasTraits):
        bar = NDArray(default_value=None, allow_none=True)
    foo = Foo(bar=[1, 2, 3])
    assert foo.bar is not None
    foo = Foo(bar=None)
    assert foo.bar is None
    foo = Foo()
    assert foo.bar is None