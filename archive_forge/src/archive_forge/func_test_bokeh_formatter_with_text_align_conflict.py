import asyncio
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models.widgets.tables import (
from packaging.version import Version
from panel.depends import bind
from panel.io.state import set_curdoc
from panel.models.tabulator import CellClickEvent, TableEditEvent
from panel.tests.util import mpl_available, serve_and_request, wait_until
from panel.util import BOKEH_JS_NAT
from panel.widgets import Button, TextInput
from panel.widgets.tables import DataFrame, Tabulator
@pytest.mark.parametrize('align', [{'x': 'right'}, 'right'], ids=['dict', 'str'])
def test_bokeh_formatter_with_text_align_conflict(align):
    data = pd.DataFrame({'x': [1.1, 2.0, 3.47]})
    formatters = {'x': NumberFormatter(format='0.0', text_align='center')}
    model = Tabulator(data, formatters=formatters, text_align=align)
    msg = "The 'text_align' in Tabulator\\.formatters\\['x'\\] is overridden by Tabulator\\.text_align"
    with pytest.warns(RuntimeWarning, match=msg):
        columns = model._get_column_definitions('x', data)
    output = columns[0].formatter.text_align
    assert output == 'right'