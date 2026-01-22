from __future__ import annotations
import asyncio
import inspect
import itertools
import json
import os
import sys
import textwrap
import types
from collections import defaultdict, namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial
from typing import (
import param
from param.parameterized import (
from param.reactive import rx
from .config import config
from .io import state
from .layout import (
from .pane import DataFrame as DataFramePane
from .pane.base import PaneBase, ReplacementPane
from .reactive import Reactive
from .util import (
from .util.checks import is_dataframe, is_mpl_axes, is_series
from .viewable import Layoutable, Viewable
from .widgets import (
from .widgets.button import _ButtonBase
def update_pane(change, parameter=pname, toggle=toggle):
    """Adds or removes subpanel from layout"""
    layout = self._expand_layout
    existing = [p for p in layout.objects if isinstance(p, Param) and p.object is change.old]
    if toggle:
        toggle.disabled = not is_parameterized(change.new)
    if not existing:
        return
    elif is_parameterized(change.new):
        parameterized = change.new
        kwargs = {k: v for k, v in self.param.values().items() if k not in ['name', 'object', 'parameters']}
        pane = Param(parameterized, name=parameterized.name, **kwargs)
        layout[layout.objects.index(existing[0])] = pane
    else:
        layout.remove(existing[0])