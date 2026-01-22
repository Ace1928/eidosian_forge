import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_widget_dtype_errors():

    class Foo(HasTraits):
        bar = DataUnion(dtype=np.uint8)
    raw_data = 100 * np.random.random((4, 4))
    w = NDArrayWidget(raw_data)
    with pytest.raises(TraitError):
        foo = Foo(bar=w)
    w.array = raw_data.astype(np.uint8)
    foo = Foo(bar=w)