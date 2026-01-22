from __future__ import annotations
import json
import sys
from collections import defaultdict
from typing import (
import param
from bokeh.models import CustomJS
from pyviz_comms import JupyterComm
from ..util import lazy_load
from ..viewable import Viewable
from .base import ModelPane
def setup_js_callbacks(root_view, root_model):
    if 'panel.models.echarts' not in sys.modules:
        return
    ref = root_model.ref['id']
    for pane in root_view.select(ECharts):
        if ref in pane._models:
            pane._models[ref][0].js_events = pane._get_js_events(ref)