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
def test_dataframe_widget_no_show_index(dataframe, document, comm):
    table = DataFrame(dataframe, show_index=False)
    model = table.get_root(document, comm)
    assert len(model.columns) == 3
    int_col, float_col, str_col = model.columns
    assert int_col.title == 'int'
    assert float_col.title == 'float'
    assert str_col.title == 'str'
    table.show_index = True
    assert len(model.columns) == 4
    index_col, int_col, float_col, str_col = model.columns
    assert index_col.title == 'index'
    assert int_col.title == 'int'
    assert float_col.title == 'float'
    assert str_col.title == 'str'