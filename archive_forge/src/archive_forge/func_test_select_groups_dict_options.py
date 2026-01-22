import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_groups_dict_options(document, comm):
    groups = dict(A=dict(a=1, b=2), B=dict(c=3))
    select = Select(value=groups['A']['a'], groups=groups, name='Select')
    widget = select.get_root(document, comm=comm)
    assert isinstance(widget, select._widget_type)
    assert widget.title == 'Select'
    assert widget.value == str(groups['A']['a'])
    assert widget.options == {'A': [('1', 'a'), ('2', 'b')], 'B': [('3', 'c')]}
    select._process_events({'value': str(groups['B']['c'])})
    assert select.value == groups['B']['c']
    select._process_events({'value': str(groups['A']['b'])})
    assert select.value == groups['A']['b']
    select.value = groups['A']['a']
    assert widget.value == str(groups['A']['a'])