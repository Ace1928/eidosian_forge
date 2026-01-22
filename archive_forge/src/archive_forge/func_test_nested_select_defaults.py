import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_defaults(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    select = NestedSelect(options=options)
    assert select.value == {2: 1000, 0: 'Andrew', 1: 'temp'}
    assert select.options == options
    assert select.levels == []
    assert select._max_depth == 3