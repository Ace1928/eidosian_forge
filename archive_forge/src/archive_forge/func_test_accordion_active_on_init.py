import pytest
from bokeh.models import Column as BkColumn, Div
import panel as pn
from panel.layout import Accordion
from panel.models import Card
def test_accordion_active_on_init(document, comm):
    combinations = [[0], [1], [0, 1]]
    for combination in combinations:
        accordion = Accordion('1', '2', active=combination)
        accordion.get_root(document, comm=comm)
        assert accordion.active == combination