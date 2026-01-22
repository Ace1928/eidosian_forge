from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
def test_tabulator_patching_no_event(page, df_mixed):
    widget = Tabulator(df_mixed)
    events = []
    widget.param.watch(lambda e: events.append(e), 'value')
    serve_component(page, widget)
    new_vals = {'str': ['AA', 'BB']}
    widget.patch({'str': [(0, new_vals['str'][0]), (1, new_vals['str'][1])]}, as_index=False)
    for v in new_vals:
        expect(page.locator(f'text="{v}"')).to_have_count(1)
    assert list(widget.value['str'].iloc[[0, 1]]) == new_vals['str']
    assert df_mixed.equals(widget.value)
    assert len(events) == 0