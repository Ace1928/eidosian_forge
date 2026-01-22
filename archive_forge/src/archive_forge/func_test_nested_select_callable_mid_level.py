import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_nested_select_callable_mid_level(document, comm):

    def list_options(level, value):
        if level == 'level_type':
            options = {f'{value['time_step']}_upper': list_options, f'{value['time_step']}_lower': list_options}
        else:
            options = [f'{value['level_type']}.json', f'{value['level_type']}.csv']
        return options
    select = NestedSelect(options={'Daily': list_options, 'Monthly': list_options}, levels=['time_step', 'level_type', 'file'])
    assert select._widgets[0].options == ['Daily', 'Monthly']
    assert select._widgets[1].options == ['Daily_upper', 'Daily_lower']
    assert select._widgets[2].options == ['Daily_upper.json', 'Daily_upper.csv']
    assert select.value == {'time_step': 'Daily', 'level_type': 'Daily_upper', 'file': 'Daily_upper.json'}
    assert select._max_depth == 3
    select.value = {'time_step': 'Monthly'}
    assert select._widgets[0].options == ['Daily', 'Monthly']
    assert select._widgets[1].options == ['Monthly_upper', 'Monthly_lower']
    assert select._widgets[2].options == ['Monthly_upper.json', 'Monthly_upper.csv']
    assert select.value == {'time_step': 'Monthly', 'level_type': 'Monthly_upper', 'file': 'Monthly_upper.json'}
    assert select._max_depth == 3