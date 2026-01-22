import pytest
import numpy as np
from traitlets import TraitError, Undefined
from ..ndarray.traits import shape_constraints
from ..ndarray.widgets import (
def test_source_must_implement_shape():
    w = NDArraySource()
    with pytest.raises(NotImplementedError):
        w.shape
    with pytest.raises(NotImplementedError):
        w.dtype