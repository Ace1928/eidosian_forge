import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_constructor_with_named_objects(document, comm):
    div1, div2 = (Div(), Div())
    p1 = pn.panel(div1, name='Div1')
    p2 = pn.panel(div2, name='Div2')
    accordion = Accordion(('Tab1', p1), ('Tab2', p2))
    model = accordion.get_root(document, comm=comm)
    assert isinstance(model, BkColumn)
    assert len(model.children) == 2
    assert all((isinstance(c, Card) for c in model.children))
    card1, card2 = model.children
    assert card1.children[0].children[0].text == '&lt;h3&gt;Tab1&lt;/h3&gt;'
    assert card1.children[1] is div1
    assert card2.children[0].children[0].text == '&lt;h3&gt;Tab2&lt;/h3&gt;'
    assert card2.children[1] is div2