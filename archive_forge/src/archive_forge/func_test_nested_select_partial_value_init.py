import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_partial_value_init(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}, 'Ben': {'temp': [500, 300], 'windspeed': [700, 500, 300]}}
    levels = ['Name', 'Var', 'Level']
    select = NestedSelect(options=options, levels=levels, value={'Name': 'Ben'})
    assert select._widgets[0].value == 'Ben'
    assert select._widgets[1].value == 'temp'
    assert select._widgets[2].value == 500
    assert select._widgets[0].visible
    assert select._widgets[1].visible
    assert select._widgets[2].visible
    assert select.value == {'Name': 'Ben', 'Var': 'temp', 'Level': 500}