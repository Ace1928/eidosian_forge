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
def link_param_method(root_view, root_model):
    """
    This preprocessor jslinks ParamMethod loading parameters to any
    widgets generated from those parameters ensuring that the loading
    indicator is enabled client side.
    """
    methods = root_view.select(lambda p: isinstance(p, ParamMethod) and p.loading_indicator)
    widgets = root_view.select(lambda w: isinstance(w, Widget) and getattr(w, '_param_pane', None) is not None)
    for widget in widgets:
        for method in methods:
            for cb in method._internal_callbacks:
                pobj = cb.cls if cb.inst is None else cb.inst
                if widget._param_pane.object is pobj and widget._param_name in cb.parameter_names:
                    if isinstance(widget, DiscreteSlider):
                        w = widget._slider
                    else:
                        w = widget
                    if 'value' in w._linkable_params:
                        w.jslink(method._inner_layout, value='loading')