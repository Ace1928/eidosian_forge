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
def test_server_cell_click_async_event():
    df = makeMixedDataFrame()
    table = Tabulator(df)
    counts = []

    async def cb(event, count=[0]):
        count[0] += 1
        counts.append(count[0])
        await asyncio.sleep(1)
        count[0] -= 1
    table.on_click(cb)
    serve_and_request(table)
    wait_until(lambda: bool(table._models))
    doc = list(table._models.values())[0][0].document
    data = df.reset_index()
    with set_curdoc(doc):
        for col in data.columns:
            for row in range(len(data)):
                event = CellClickEvent(model=None, column=col, row=row)
                table._process_event(event)
    wait_until(lambda: len(counts) >= 1 and max(counts) > 1)