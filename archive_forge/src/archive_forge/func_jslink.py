from __future__ import annotations
import sys
from collections import defaultdict
from functools import partial
from typing import (
import param
from bokeh.models import Range1d, Spacer as _BkSpacer
from bokeh.themes.theme import Theme
from packaging.version import Version
from param.parameterized import register_reference_transform
from param.reactive import bind
from ..io import state, unlocked
from ..layout import (
from ..viewable import Layoutable, Viewable
from ..widgets import Player
from .base import PaneBase, RerenderError, panel
from .plot import Bokeh, Matplotlib
from .plotly import Plotly
def jslink(self, target, code=None, args=None, bidirectional=False, **links):
    if links and code:
        raise ValueError('Either supply a set of properties to link as keywords or a set of JS code callbacks, not both.')
    elif not links and (not code):
        raise ValueError('Declare parameters to link or a set of callbacks, neither was defined.')
    if args is None:
        args = {}
    from ..links import Link
    return Link(self, target, properties=links, code=code, args=args, bidirectional=bidirectional)