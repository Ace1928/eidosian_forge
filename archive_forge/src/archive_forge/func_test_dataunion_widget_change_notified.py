import pytest
import numpy as np
from traitlets import HasTraits, TraitError, observe
from ipywidgets import Widget
from ..ndarray.traits import shape_constraints
from ..ndarray.union import DataUnion, get_union_array
from ..ndarray.widgets import NDArrayWidget, NDArraySource
def test_dataunion_widget_change_notified(mock_comm):
    ns = {'counter': 0}

    class Foo(Widget):
        bar = DataUnion().tag(sync=True)

        @observe('bar')
        def on_bar_change(self, change):
            ns['counter'] += 1
    raw_data = np.ones((4, 4))
    raw_data2 = np.ones((4, 4, 2))
    w = NDArrayWidget(raw_data)
    foo = Foo(bar=w)
    foo.comm = mock_comm
    assert ns['counter'] == 1
    w.array = raw_data2
    assert ns['counter'] == 2
    foo.bar = raw_data
    assert ns['counter'] == 3
    assert len(mock_comm.log_send) == 2
    foo = Foo(bar=raw_data)
    assert ns['counter'] == 4
    foo.bar = w
    assert ns['counter'] == 5