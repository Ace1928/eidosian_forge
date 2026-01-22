import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_cross_select_move_unselected_to_selected():
    cross_select = CrossSelector(options=['A', 'B', 'C', 1, 2, 3], value=['A', 1], size=5)
    cross_select._lists[False].value = ['B', '3']
    cross_select._buttons[True].clicks = 1
    assert cross_select.value == ['A', 1, 'B', 3]
    assert cross_select._lists[True].options == ['A', 'B', '1', '3']