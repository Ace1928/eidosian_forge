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
def get_breadths(node, graph, depth=0, breadths=None):
    if breadths is None:
        breadths = defaultdict(list)
        breadths[depth].append(node)
    for sub in graph.get(node, []):
        if sub not in breadths[depth + 1]:
            breadths[depth + 1].append(sub)
        get_breadths(sub, graph, depth + 1, breadths)
    return breadths