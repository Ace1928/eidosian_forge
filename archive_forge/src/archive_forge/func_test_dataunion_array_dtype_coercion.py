import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_array_dtype_coercion():

    class Foo(HasTraits):
        bar = DataUnion(dtype=np.uint8)
    raw_data = 100 * np.random.random((4, 4))
    with pytest.warns(UserWarning) as warnings:
        foo = Foo(bar=raw_data)
        assert len(warnings) == 1
        assert 'Given trait value dtype "float64" does not match required type "uint8"' in str(warnings[0].message)
    assert not np.array_equiv(foo.bar, raw_data)
    assert np.array_equiv(foo.bar, raw_data.astype(np.uint8))