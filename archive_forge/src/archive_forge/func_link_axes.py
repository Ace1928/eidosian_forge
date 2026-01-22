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
def link_axes(root_view, root_model):
    """
    Pre-processing hook to allow linking axes across HoloViews bokeh
    plots.
    """
    panes = root_view.select(HoloViews)
    if not panes:
        return
    from holoviews.core.options import Store
    from holoviews.core.util import max_range, unique_iterator
    from holoviews.plotting.bokeh.element import ElementPlot
    ref = root_model.ref['id']
    range_map = defaultdict(list)
    for pane in panes:
        if ref not in pane._plots:
            continue
        plot = pane._plots[ref][0]
        if not pane.linked_axes or plot.renderer.backend != 'bokeh' or (not getattr(plot, 'shared_axes', False)):
            continue
        for p in plot.traverse(specs=[ElementPlot]):
            if p.current_frame is None:
                continue
            axiswise = Store.lookup_options('bokeh', p.current_frame, 'norm').kwargs.get('axiswise')
            if not p.shared_axes or axiswise:
                continue
            fig = p.state
            if fig.x_range.tags:
                tag = tuple((tuple(t) if isinstance(t, list) else t for t in fig.x_range.tags[0]))
                range_map[tag].append((fig, p, fig.xaxis[0], fig.x_range))
            if fig.y_range.tags:
                tag = tuple((tuple(t) if isinstance(t, list) else t for t in fig.y_range.tags[0]))
                range_map[tag].append((fig, p, fig.yaxis[0], fig.y_range))
    for tag, axes in range_map.items():
        fig, p, ax, axis = axes[0]
        if isinstance(axis, Range1d):
            start, end = max_range([(ax[-1].start, ax[-1].end) for ax in axes if isinstance(ax[-1], Range1d)])
            if axis.start > axis.end:
                end, start = (start, end)
            axis.start = start
            axis.end = end
        for fig, p, pax, _ in axes[1:]:
            changed = []
            if type(ax) is not type(pax):
                continue
            if tag in fig.x_range.tags and axis is not fig.x_range:
                if hasattr(axis, 'factors'):
                    axis.factors = list(unique_iterator(axis.factors + fig.x_range.factors))
                fig.x_range = axis
                p.handles['x_range'] = axis
                changed.append('x_range')
            if tag in fig.y_range.tags and axis is not fig.y_range:
                if hasattr(axis, 'factors'):
                    axis.factors = list(unique_iterator(axis.factors + fig.y_range.factors))
                fig.y_range = axis
                p.handles['y_range'] = axis
                changed.append('y_range')
            subplots = p.subplots
            if subplots:
                plots = subplots.values()
            else:
                plots = [p]
            for sp in plots:
                for callback in sp.callbacks:
                    models = callback.models
                    if hasattr(callback, 'extra_models'):
                        models += callback.extra_models
                    if not any((c in models for c in changed)):
                        continue
                    if 'x_range' in changed:
                        sp.handles['x_range'] = p.handles['x_range']
                    if 'y_range' in changed:
                        sp.handles['y_range'] = p.handles['y_range']
                    callback.reset()
                    callback.initialize(plot_id=p.id)