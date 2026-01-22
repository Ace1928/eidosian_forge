from bokeh.events import ButtonClick, MenuItemClick
from panel.widgets import Button, MenuButton, Toggle
def test_button(document, comm):
    button = Button(name='Button')
    widget = button.get_root(document, comm=comm)
    assert isinstance(widget, button._widget_type)
    assert widget.label == 'Button'
    button._process_event(None)
    assert button.clicks == 1