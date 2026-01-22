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
def test_tabulator_cell_click_event_error_duplicate_index():
    df = pd.DataFrame(data={'A': [1, 2]}, index=['a', 'a'])
    table = Tabulator(df, sorters=[{'field': 'A', 'sorter': 'number', 'dir': 'desc'}])
    values = []
    table.on_click(lambda e: values.append((e.column, e.row, e.value)))
    event = CellClickEvent(model=None, column='y', row=0)
    with pytest.raises(ValueError, match="Found this duplicate index: 'a'"):
        table._process_event(event)