from bokeh.events import ButtonClick, MenuItemClick
from panel.widgets import Button, MenuButton, Toggle
def test_menu_button(document, comm):
    menu_items = [('Option A', 'a'), ('Option B', 'b'), ('Option C', 'c'), None, ('Help', 'help')]
    menu_button = MenuButton(items=menu_items)
    widget = menu_button.get_root(document, comm=comm)
    events = []

    def callback(event):
        events.append(event.new)
    menu_button.param.watch(callback, 'clicked')
    menu_button._process_event(MenuItemClick(widget, 'b'))
    assert events == ['b']