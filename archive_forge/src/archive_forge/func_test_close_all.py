import inspect
import pytest
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display
from IPython.utils.capture import capture_output
from .. import widget
from ..widget import Widget
from ..widget_button import Button
import copy
def test_close_all():
    widgets = [Button() for i in range(10)]
    assert len(widget._instances) > 0, 'expect active widgets'
    assert widget._instances[widgets[0].model_id] is widgets[0]
    Widget.close_all()
    assert len(widget._instances) == 0, 'active widgets should be cleared'