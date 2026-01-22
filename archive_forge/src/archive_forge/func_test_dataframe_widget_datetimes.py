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
def test_dataframe_widget_datetimes(document, comm):
    df = pd.DataFrame({'int': [1, 2, 3]}, index=pd.date_range('2000-01-01', periods=3))
    table = DataFrame(df)
    model = table.get_root(document, comm)
    dt_col, _ = model.columns
    assert dt_col.title == 'index'
    assert isinstance(dt_col.formatter, DateFormatter)
    assert isinstance(dt_col.editor, CellEditor)