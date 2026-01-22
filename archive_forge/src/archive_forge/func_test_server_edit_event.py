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
def test_server_edit_event():
    df = makeMixedDataFrame()
    table = Tabulator(df)
    serve_and_request(table)
    wait_until(lambda: bool(table._models))
    ref, (model, _) = list(table._models.items())[0]
    doc = list(table._documents.keys())[0]
    events = []
    table.on_edit(lambda e: events.append(e))
    new_data = dict(model.source.data)
    new_data['B'][1] = 3.14
    table._server_change(doc, ref, None, 'data', model.source.data, new_data)
    table._server_event(doc, TableEditEvent(model, 'B', 1))
    wait_until(lambda: len(events) == 1)
    assert events[0].value == 3.14
    assert events[0].old == 1