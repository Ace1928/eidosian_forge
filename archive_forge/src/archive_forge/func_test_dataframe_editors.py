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
def test_dataframe_editors(dataframe, document, comm):
    editor = SelectEditor(options=['A', 'B', 'C'])
    table = DataFrame(dataframe, editors={'str': editor})
    model = table.get_root(document, comm)
    model_editor = model.columns[-1].editor
    assert isinstance(model_editor, SelectEditor) is not editor
    assert isinstance(model_editor, SelectEditor)
    assert model_editor.options == ['A', 'B', 'C']