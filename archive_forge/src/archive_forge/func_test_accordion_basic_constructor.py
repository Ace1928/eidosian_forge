import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_basic_constructor(document, comm):
    accordion = Accordion('plain', 'text')
    model = accordion.get_root(document, comm=comm)
    assert isinstance(model, BkColumn)
    assert len(model.children) == 2
    assert all((isinstance(c, Card) for c in model.children))
    card1, card2 = model.children
    assert 'plain' in card1.children[1].text
    assert 'text' in card2.children[1].text