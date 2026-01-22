import numpy as np
import pytest
from panel.layout import GridBox, Row
from panel.pane import panel
from panel.tests.util import mpl_available
from panel.widgets import (
def test_select_parameterized_option_labels():
    c1 = panel('Value1', name='V1')
    c2 = panel('Value2')
    c3 = panel('Value3', name='V3')
    select = Select(options=[c1, c2, c3], value=c1)
    assert select.labels == ['V1', 'Markdown(str)', 'V3']