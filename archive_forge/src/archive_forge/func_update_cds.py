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
def update_cds(new, nb=False):
    sid = str(new.args[0])
    if sid not in sessions:
        sessions.append(sid)
    if 'finished processing events' in new.msg:
        msg = new.getMessage()
        etype = 'processing'
        try:
            index = cds.data['msg'].index(msg.replace('finished processing', 'received'))
        except Exception:
            return
        patch = {'x1': [(index, new.created * 1000)], 'color': [(index, EVENT_TYPES[etype])], 'type': [(index, etype)], 'msg': [(index, msg.replace('finished processing', 'processed'))]}
        cds.patch(patch)
    elif new.msg == LOG_SESSION_CREATED:
        index = cds.data['msg'].index(LOG_SESSION_LAUNCHING % sid)
        etype = 'initializing'
        patch = {'x1': [(index, new.created * 1000)], 'color': [(index, EVENT_TYPES[etype])], 'type': [(index, etype)], 'msg': [(index, f'Session {sid} initializing')]}
        cds.patch(patch)
    elif 'finished executing periodic callback' in new.msg:
        etype = 'periodic'
        msg = new.getMessage()
        index = cds.data['msg'].index(msg.replace('finished executing', 'executing'))
        patch = {'x1': [(index, new.created * 1000)], 'color': [(index, EVENT_TYPES[etype])], 'type': [(index, etype)]}
        cds.patch(patch)
    elif new.msg.endswith('rendered'):
        try:
            index = cds.data['msg'].index(f'Session {sid} initializing')
            x0 = cds.data['x1'][index]
        except ValueError:
            x0 = new.created * 1000
        etype = 'rendering'
        event = {'x0': [x0], 'x1': [new.created * 1000], 'y0': [(sid, -0.25)], 'y1': [(sid, 0.25)], 'session': [sid], 'msg': [new.getMessage().replace('rendered', 'rendering')], 'color': [EVENT_TYPES[etype]], 'line_color': ['black'], 'type': [etype]}
        cds.stream(event)
    else:
        msg = new.getMessage()
        line_color = 'black'
        if msg.startswith('Session %s logged' % sid):
            etype = 'logging'
            line_color = EVENT_TYPES.get(etype)
        elif msg.startswith(LOG_SESSION_DESTROYED % sid):
            etype = 'destroyed'
            line_color = EVENT_TYPES.get(etype)
        elif 'executing periodic callback' in msg:
            etype = 'periodic'
            line_color = EVENT_TYPES.get(etype)
        else:
            etype = 'processing'
        event = {'x0': [new.created * 1000], 'x1': [new.created * 1000], 'y0': [(sid, -0.25)], 'y1': [(sid, 0.25)], 'session': [sid], 'msg': [msg], 'color': [EVENT_TYPES[etype]], 'line_color': [line_color], 'type': [etype]}
        if p.y_range.factors != sessions:
            p.y_range.factors = list(sessions)
        cds.stream(event)
    if nb:
        push_notebook(bk_pane)