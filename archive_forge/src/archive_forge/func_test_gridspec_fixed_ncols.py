import pytest
from bokeh.models import Div
from panel.depends import depends
from panel.layout import GridBox, GridSpec, Spacer
from panel.widgets import IntSlider
def test_gridspec_fixed_ncols():
    grid = GridSpec(ncols=3)
    for index in range(5):
        grid[index, :] = 'Hello World'