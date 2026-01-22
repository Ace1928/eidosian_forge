import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_two_widget_arg_with_remaining_arg():
    widget = IntSlider(value=0)

    def add(value, value2):
        return value + value2
    bound_function = bind(add, widget)
    assert bound_function(1) == 1
    widget.value = 1
    assert bound_function(2) == 3
    assert bound_function(value2=3) == 4