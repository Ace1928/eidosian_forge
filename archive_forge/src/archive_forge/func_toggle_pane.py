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
def toggle_pane(change, parameter=pname):
    """Adds or removes subpanel from layout"""
    parameterized = getattr(self.object, parameter)
    existing = [p for p in self._expand_layout.objects if isinstance(p, Param) and p.object in recursive_parameterized(parameterized)]
    if not change.new:
        self._expand_layout[:] = [e for e in self._expand_layout.objects if e not in existing]
    elif change.new:
        kwargs = {k: v for k, v in self.param.values().items() if k not in ['name', 'object', 'parameters']}
        pane = Param(parameterized, name=parameterized.name, **kwargs)
        if isinstance(self._expand_layout, Tabs):
            title = self.object.param[parameter].label
            pane = (title, pane)
        self._expand_layout.append(pane)