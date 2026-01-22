import pytest
from panel.depends import bind, transform_reference
from panel.pane import panel
from panel.param import ParamFunction
from panel.widgets import IntSlider
def test_bind_widget_to_arg():
    widget = IntSlider(value=0)

    def add1(value):
        return value + 1
    bound_function = bind(add1, widget)
    assert bound_function() == 1
    widget.value = 1
    assert bound_function() == 2
    with pytest.raises(TypeError):
        bound_function(1)