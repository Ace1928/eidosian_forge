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
def test_tabulator_download_menu_default():
    df = makeMixedDataFrame()
    table = Tabulator(df)
    filename, button = table.download_menu()
    assert isinstance(filename, TextInput)
    assert isinstance(button, Button)
    assert filename.value == 'table.csv'
    assert filename.name == 'Filename'
    assert button.name == 'Download'