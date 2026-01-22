import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_toggle_group_check(document, comm):
    for widget_type in ToggleGroup._widgets_type:
        select = ToggleGroup(options={'A': 'A', '1': 1, 'C': object}, value=[1, object], name='CheckButtonGroup', widget_type=widget_type, behavior='check')
        widget = select.get_root(document, comm=comm)
        assert isinstance(widget, select._widget_type)
        assert widget.active == [1, 2]
        assert widget.labels == ['A', '1', 'C']
        widget.active = [2]
        select._process_events({'active': [2]})
        assert select.value == [object]
        widget.active = [0, 2]
        select._process_events({'active': [0, 2]})
        assert select.value == ['A', object]
        select.value = [object, 'A']
        assert widget.active == [2, 0]
        widget.active = []
        select._process_events({'active': []})
        assert select.value == []
        select.value = ['A', 'B']
        select.options = ['B', 'C']
        select.options = ['A', 'B']
        assert widget.labels[widget.active[0]] == 'B'