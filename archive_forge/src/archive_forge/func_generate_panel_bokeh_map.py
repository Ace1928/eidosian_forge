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
def generate_panel_bokeh_map(root_model, panel_views):
    """
    mapping panel elements to its bokeh models
    """
    map_hve_bk = defaultdict(list)
    ref = root_model.ref['id']
    for pane in panel_views:
        if root_model.ref['id'] in pane._models:
            plot, subpane = pane._plots.get(ref, (None, None))
            if plot is None:
                continue
            bk_plots = plot.traverse(lambda x: x, [is_bokeh_element_plot])
            for plot in bk_plots:
                for hv_elem in plot.link_sources:
                    map_hve_bk[hv_elem].append(plot)
    return map_hve_bk