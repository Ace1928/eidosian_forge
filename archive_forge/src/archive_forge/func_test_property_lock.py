import pytest
from unittest import mock
from traitlets import Bool, Tuple, List, Instance, CFloat, CInt, Float, Int, TraitError, observe
from .utils import setup, teardown
import ipywidgets
from ipywidgets import Widget
def test_property_lock(echo):

    class AnnoyingWidget(Widget):
        value = Float().tag(sync=True)
        stop = Bool(False)

        @observe('value')
        def _propagate_value(self, change):
            print('_propagate_value', change.new)
            if self.stop:
                return
            if change.new == 42:
                self.value = 2
            if change.new == 2:
                self.stop = True
                self.value = 42
    widget = AnnoyingWidget(value=1)
    assert widget.value == 1
    widget._send = mock.MagicMock()
    widget.set_state({'value': 42})
    assert widget.value == 42
    assert widget.stop is True
    calls = []
    widget._send.assert_has_calls(calls)