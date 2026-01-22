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
def test_dataframe_triggers(dataframe):
    events = []

    def increment(event, events=events):
        events.append(event)
    table = DataFrame(dataframe)
    table.param.watch(increment, 'value')
    table._process_events({'data': {'str': ['C', 'B', 'A']}})
    assert len(events) == 1