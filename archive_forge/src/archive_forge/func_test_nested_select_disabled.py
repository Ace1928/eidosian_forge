import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_disabled(document, comm):
    options = {'Andrew': {'temp': [1000, 925, 700, 500, 300], 'vorticity': [500, 300]}}
    select = NestedSelect(options=options, levels=['Name', 'Var', 'Level'])
    select.disabled = True
    assert select._widgets[0].disabled
    select.disabled = False
    assert not select._widgets[0].disabled