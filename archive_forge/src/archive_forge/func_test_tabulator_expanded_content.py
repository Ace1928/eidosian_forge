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
def test_tabulator_expanded_content(document, comm):
    df = makeMixedDataFrame()
    table = Tabulator(df, expanded=[0, 1], row_content=lambda r: r.A)
    model = table.get_root(document, comm)
    assert len(model.children) == 2
    assert 0 in model.children
    row0 = model.children[0]
    assert row0.text == '&lt;pre&gt;0.0&lt;/pre&gt;'
    assert 1 in model.children
    row1 = model.children[1]
    assert row1.text == '&lt;pre&gt;1.0&lt;/pre&gt;'
    table.expanded = [1, 2]
    assert 0 not in model.children
    assert 1 in model.children
    assert row1 is model.children[1]
    assert 2 in model.children
    row2 = model.children[2]
    assert row2.text == '&lt;pre&gt;2.0&lt;/pre&gt;'