import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_dynamic_levels(document, comm):
    select = NestedSelect(options={'Easy': {'Easy_A': {}, 'Easy_B': {}}, 'Medium': {'Medium_A': {}, 'Medium_B': {'Medium_B_1': []}, 'Medium_C': {'Medium_C_1': ['Medium_C_1_1'], 'Medium_C_2': ['Medium_C_2_1', 'Medium_C_2_2']}}, 'Hard': {}}, levels=['A', 'B', 'C', 'D'])
    assert select._widgets[0].visible
    assert select._widgets[1].visible
    assert not select._widgets[2].visible
    assert not select._widgets[3].visible
    assert select._widgets[0].options == ['Easy', 'Medium', 'Hard']
    assert select._widgets[1].options == ['Easy_A', 'Easy_B']
    assert select._widgets[2].options == []
    assert select._widgets[3].options == []
    assert select.value == {'A': 'Easy', 'B': 'Easy_A', 'C': None, 'D': None}
    select.value = {'A': 'Medium', 'B': 'Medium_C'}
    assert select._widgets[0].visible
    assert select._widgets[1].visible
    assert select._widgets[2].visible
    assert select._widgets[3].visible
    assert select._widgets[0].options == ['Easy', 'Medium', 'Hard']
    assert select._widgets[1].options == ['Medium_A', 'Medium_B', 'Medium_C']
    assert select._widgets[2].options == ['Medium_C_1', 'Medium_C_2']
    assert select._widgets[3].options == ['Medium_C_1_1']
    assert select.value == {'A': 'Medium', 'B': 'Medium_C', 'C': 'Medium_C_1', 'D': 'Medium_C_1_1'}
    select.value = {'A': 'Hard'}
    assert select._widgets[0].visible
    assert not select._widgets[1].visible
    assert not select._widgets[2].visible
    assert not select._widgets[3].visible
    assert select._widgets[0].options == ['Easy', 'Medium', 'Hard']
    assert select._widgets[1].options == []
    assert select._widgets[2].options == []
    assert select._widgets[3].options == []
    assert select.value == {'A': 'Hard', 'B': None, 'C': None, 'D': None}