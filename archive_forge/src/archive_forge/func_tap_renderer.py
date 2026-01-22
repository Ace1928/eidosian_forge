from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def tap_renderer(plot, element):
    from bokeh.models import TapTool
    gr = plot.handles['glyph_renderer']
    tap = plot.state.select_one(TapTool)
    tap.renderers = [gr]