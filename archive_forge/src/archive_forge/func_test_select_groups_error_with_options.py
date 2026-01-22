import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_groups_error_with_options():
    with pytest.raises(ValueError):
        Select(options=[1, 2], groups=dict(a=[1], b=[2]), name='Select')
    opts = [1, 2, 3]
    groups = dict(a=[1, 2], b=[3])
    select = Select(options=opts, name='Select')
    with pytest.raises(ValueError):
        select.groups = groups
    select = Select(groups=groups, name='Select')
    with pytest.raises(ValueError):
        select.options = opts