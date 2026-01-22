import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridbox_ncols(document, comm):
    grid_box = GridBox(Div(), Div(), Div(), Div(), Div(), Div(), Div(), Div(), ncols=3)
    model = grid_box.get_root(document, comm=comm)
    assert len(model.children) == 8
    coords = [(0, 0, 1, 1), (0, 1, 1, 1), (0, 2, 1, 1), (1, 0, 1, 1), (1, 1, 1, 1), (1, 2, 1, 1), (2, 0, 1, 1), (2, 1, 1, 1)]
    for child, coord in zip(model.children, coords):
        assert child[1:] == coord