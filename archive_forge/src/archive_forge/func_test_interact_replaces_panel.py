from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_interact_replaces_panel(document, comm):

    def test(a):
        return a if a else BkDiv(text='Test')
    interact_pane = interactive(test, a=False)
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.Checkbox)
    assert widget.value == False
    column = interact_pane.layout.get_root(document, comm=comm)
    assert isinstance(column, BkColumn)
    div = column.children[1].children[0]
    assert div.text == 'Test'
    widget.value = True
    div = column.children[1].children[0]
    assert div.text == '&lt;pre&gt;True&lt;/pre&gt;'