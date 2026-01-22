import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_source_validation():

    class Source(NDArraySource):

        def _get_dtype(self):
            return 'uint8'

        def _get_shape(self):
            return [2, 3]
    w = Source()

    class Foo(Widget):
        bar = DataUnion(dtype='uint8', shape_constraint=shape_constraints(2, 3))
    foo = Foo(bar=w)