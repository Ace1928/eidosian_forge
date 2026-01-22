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
@classmethod
def is_pyecharts(cls, obj):
    if 'pyecharts' in sys.modules:
        import pyecharts
        return isinstance(obj, pyecharts.charts.chart.Chart)
    return False