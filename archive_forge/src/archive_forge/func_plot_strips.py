from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
def plot_strips(self, jitter, dodge, color, plot_kws):
    width = 0.8 * self._native_width
    offsets = self._nested_offsets(width, dodge)
    if jitter is True:
        jlim = 0.1
    else:
        jlim = float(jitter)
    if 'hue' in self.variables and dodge and (self._hue_map.levels is not None):
        jlim /= len(self._hue_map.levels)
    jlim *= self._native_width
    jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)
    iter_vars = [self.orient]
    if dodge:
        iter_vars.append('hue')
    ax = self.ax
    dodge_move = jitter_move = 0
    if 'marker' in plot_kws and (not MarkerStyle(plot_kws['marker']).is_filled()):
        plot_kws.pop('edgecolor', None)
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=True):
        ax = self._get_axes(sub_vars)
        if offsets is not None and (offsets != 0).any():
            dodge_move = offsets[sub_data['hue'].map(self._hue_map.levels.index)]
        jitter_move = jitterer(size=len(sub_data)) if len(sub_data) > 1 else 0
        adjusted_data = sub_data[self.orient] + dodge_move + jitter_move
        sub_data[self.orient] = adjusted_data
        self._invert_scale(ax, sub_data)
        points = ax.scatter(sub_data['x'], sub_data['y'], color=color, **plot_kws)
        if 'hue' in self.variables:
            points.set_facecolors(self._hue_map(sub_data['hue']))
    self._configure_legend(ax, _scatter_legend_artist, common_kws=plot_kws)