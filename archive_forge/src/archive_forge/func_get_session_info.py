import datetime as dt
import logging
import os
import sys
import time
from functools import partial
import bokeh
import numpy as np
import pandas as pd
import param
from bokeh.models import HoverTool
from bokeh.plotting import ColumnDataSource, figure
from ..config import config, panel_extension as extension
from ..depends import bind
from ..layout import (
from ..pane import HTML, Bokeh
from ..template import FastListTemplate
from ..widgets import (
from ..widgets.indicators import Trend
from .logging import (
from .notebook import push_notebook
from .profile import profiling_tabs
from .server import set_curdoc
from .state import state
def get_session_info(doc=None):
    df = get_session_data()
    total = Trend(data=df[['time', 'total']], plot_x='time', plot_y='total', plot_type='step', name='Total Sessions', width=300, height=300)
    active = Trend(data=df[['time', 'live']], plot_x='time', plot_y='live', plot_type='step', name='Active Sessions', width=300, height=300)
    render = Trend(data=df[['time', 'render']], plot_x='time', plot_y='render', plot_type='step', name='Avg. Time to Render (s)', width=300, height=300)
    duration = Trend(data=df[['time', 'duration']], plot_x='time', plot_y='duration', plot_type='step', name='Avg. Session Duration (s)', width=300, height=300)

    def update_session_info(event):
        df = get_session_data()
        for trend in (total, active, render, duration):
            trend.data = df[[trend.plot_x, trend.plot_y]]
    watcher = state.param.watch(update_session_info, 'session_info')
    if doc:

        def _unwatch_session_info(session_context):
            state.param.unwatch(watcher)
        doc.on_session_destroyed(_unwatch_session_info)
    return (total, active, render, duration)