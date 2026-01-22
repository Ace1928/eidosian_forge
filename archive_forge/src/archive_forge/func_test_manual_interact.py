from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_manual_interact():

    def test(a):
        return a
    interact_pane = interactive(test, params={'manual_update': True}, a=False)
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.Checkbox)
    assert widget.value == False
    assert isinstance(interact_pane._widgets['manual'], widgets.Button)