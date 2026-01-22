import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_set_card_collapsed_toggle(document, comm, accordion):
    accordion.toggle = True
    accordion.get_root(document, comm=comm)
    events = []
    accordion.param.watch(lambda e: events.append(e), 'active')
    c1, c2 = accordion._panels.values()
    c1.collapsed = False
    assert accordion.active == [0]
    assert len(events) == 1
    c2.collapsed = False
    assert accordion.active == [1]
    assert len(events) == 2