from datetime import date
from bokeh.models import Column as BkColumn, Div as BkDiv
from panel import widgets
from panel.interact import interactive
from panel.models import HTML as BkHTML
from panel.pane import HTML
def test_interact_replaces_model(document, comm):

    def test(a):
        return 'ABC' if a else BkDiv(text='Test')
    interact_pane = interactive(test, a=False)
    pane = interact_pane._pane
    widget = interact_pane._widgets['a']
    assert isinstance(widget, widgets.Checkbox)
    assert widget.value == False
    column = interact_pane.layout.get_root(document, comm=comm)
    assert isinstance(column, BkColumn)
    div = column.children[1].children[0]
    assert isinstance(div, BkDiv)
    assert div.text == 'Test'
    assert pane._models[column.ref['id']][0] is div
    widget.value = True
    new_pane = interact_pane._pane
    assert new_pane is not pane
    new_div = column.children[1].children[0]
    assert isinstance(new_div, BkHTML)
    assert new_div.text.endswith('&lt;p&gt;ABC&lt;/p&gt;\n')
    assert new_pane._models[column.ref['id']][0] is new_div
    interact_pane._cleanup(column)
    assert len(interact_pane._internal_callbacks) == 5