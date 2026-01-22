from bokeh.events import ButtonClick, MenuItemClick
from panel.widgets import Button, MenuButton, Toggle
def test_toggle(document, comm):
    toggle = Toggle(name='Toggle', value=True)
    widget = toggle.get_root(document, comm=comm)
    assert isinstance(widget, toggle._widget_type)
    assert widget.active == True
    assert widget.label == 'Toggle'
    widget.active = False
    toggle._process_events({'active': widget.active})
    assert toggle.value == False
    toggle.value = True
    assert widget.active == True