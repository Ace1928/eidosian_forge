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
def plot_violins(self, width, dodge, gap, split, color, fill, linecolor, linewidth, inner, density_norm, common_norm, kde_kws, inner_kws, plot_kws):
    iter_vars = [self.orient, 'hue']
    value_var = {'x': 'y', 'y': 'x'}[self.orient]
    inner_options = ['box', 'quart', 'stick', 'point', None]
    _check_argument('inner', inner_options, inner, prefix=True)
    _check_argument('density_norm', ['area', 'count', 'width'], density_norm)
    if linewidth is None:
        if fill:
            linewidth = 1.25 * mpl.rcParams['patch.linewidth']
        else:
            linewidth = mpl.rcParams['lines.linewidth']
    if inner is not None and inner.startswith('box'):
        box_width = inner_kws.pop('box_width', linewidth * 4.5)
        whis_width = inner_kws.pop('whis_width', box_width / 3)
        marker = inner_kws.pop('marker', '_' if self.orient == 'x' else '|')
    kde = KDE(**kde_kws)
    ax = self.ax
    violin_data = []
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
        sub_data['weight'] = sub_data.get('weights', 1)
        stat_data = kde._transform(sub_data, value_var, [])
        maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
        if not fill:
            linecolor = maincolor
            maincolor = 'none'
        default_kws = dict(facecolor=maincolor, edgecolor=linecolor, linewidth=linewidth)
        violin_data.append({'position': sub_vars[self.orient], 'observations': sub_data[value_var], 'density': stat_data['density'], 'support': stat_data[value_var], 'kwargs': {**default_kws, **plot_kws}, 'sub_vars': sub_vars, 'ax': self._get_axes(sub_vars)})

    def vars_to_key(sub_vars):
        return tuple(((k, v) for k, v in sub_vars.items() if k != self.orient))
    norm_keys = [vars_to_key(violin['sub_vars']) for violin in violin_data]
    if common_norm:
        common_max_density = np.nanmax([v['density'].max() for v in violin_data])
        common_max_count = np.nanmax([len(v['observations']) for v in violin_data])
        max_density = {key: common_max_density for key in norm_keys}
        max_count = {key: common_max_count for key in norm_keys}
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN (slice|axis) encountered')
            max_density = {key: np.nanmax([v['density'].max() for v in violin_data if vars_to_key(v['sub_vars']) == key]) for key in norm_keys}
        max_count = {key: np.nanmax([len(v['observations']) for v in violin_data if vars_to_key(v['sub_vars']) == key]) for key in norm_keys}
    real_width = width * self._native_width
    for violin in violin_data:
        index = pd.RangeIndex(0, max(len(violin['support']), 1))
        data = pd.DataFrame({self.orient: violin['position'], value_var: violin['support'], 'density': violin['density'], 'width': real_width}, index=index)
        if dodge:
            self._dodge(violin['sub_vars'], data)
        if gap:
            data['width'] *= 1 - gap
        norm_key = vars_to_key(violin['sub_vars'])
        hw = data['width'] / 2
        peak_density = violin['density'].max()
        if np.isnan(peak_density):
            span = 1
        elif density_norm == 'area':
            span = data['density'] / max_density[norm_key]
        elif density_norm == 'count':
            count = len(violin['observations'])
            span = data['density'] / peak_density * (count / max_count[norm_key])
        elif density_norm == 'width':
            span = data['density'] / peak_density
        span = span * hw * (2 if split else 1)
        right_side = 0 if 'hue' not in self.variables else self._hue_map.levels.index(violin['sub_vars']['hue']) % 2
        if split:
            offsets = (hw, span - hw) if right_side else (span - hw, hw)
        else:
            offsets = (span, span)
        ax = violin['ax']
        _, invx = _get_transform_functions(ax, 'x')
        _, invy = _get_transform_functions(ax, 'y')
        inv_pos = {'x': invx, 'y': invy}[self.orient]
        inv_val = {'x': invx, 'y': invy}[value_var]
        linecolor = violin['kwargs']['edgecolor']
        if np.isnan(peak_density):
            pos = data[self.orient].iloc[0]
            val = violin['observations'].mean()
            if self.orient == 'x':
                x, y = ([pos - offsets[0], pos + offsets[1]], [val, val])
            else:
                x, y = ([val, val], [pos - offsets[0], pos + offsets[1]])
            ax.plot(invx(x), invy(y), color=linecolor, linewidth=linewidth)
            continue
        plot_func = {'x': ax.fill_betweenx, 'y': ax.fill_between}[self.orient]
        plot_func(inv_val(data[value_var]), inv_pos(data[self.orient] - offsets[0]), inv_pos(data[self.orient] + offsets[1]), **violin['kwargs'])
        obs = violin['observations']
        pos_dict = {self.orient: violin['position'], 'width': real_width}
        if dodge:
            self._dodge(violin['sub_vars'], pos_dict)
        if gap:
            pos_dict['width'] *= 1 - gap
        if inner is None:
            continue
        elif inner.startswith('point'):
            pos = np.array([pos_dict[self.orient]] * len(obs))
            if split:
                pos += (-1 if right_side else 1) * pos_dict['width'] / 2
            x, y = (pos, obs) if self.orient == 'x' else (obs, pos)
            kws = {'color': linecolor, 'edgecolor': linecolor, 's': (linewidth * 2) ** 2, 'zorder': violin['kwargs'].get('zorder', 2) + 1, **inner_kws}
            ax.scatter(invx(x), invy(y), **kws)
        elif inner.startswith('stick'):
            pos0 = np.interp(obs, data[value_var], data[self.orient] - offsets[0])
            pos1 = np.interp(obs, data[value_var], data[self.orient] + offsets[1])
            pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
            val_pts = np.stack([inv_val(obs), inv_val(obs)])
            segments = np.stack([pos_pts, val_pts]).transpose(2, 1, 0)
            if self.orient == 'y':
                segments = segments[:, :, ::-1]
            kws = {'color': linecolor, 'linewidth': linewidth / 2, **inner_kws}
            lines = mpl.collections.LineCollection(segments, **kws)
            ax.add_collection(lines, autolim=False)
        elif inner.startswith('quart'):
            stats = np.percentile(obs, [25, 50, 75])
            pos0 = np.interp(stats, data[value_var], data[self.orient] - offsets[0])
            pos1 = np.interp(stats, data[value_var], data[self.orient] + offsets[1])
            pos_pts = np.stack([inv_pos(pos0), inv_pos(pos1)])
            val_pts = np.stack([inv_val(stats), inv_val(stats)])
            segments = np.stack([pos_pts, val_pts]).transpose(2, 0, 1)
            if self.orient == 'y':
                segments = segments[:, ::-1, :]
            dashes = [(1.25, 0.75), (2.5, 1), (1.25, 0.75)]
            for i, segment in enumerate(segments):
                kws = {'color': linecolor, 'linewidth': linewidth, 'dashes': dashes[i], **inner_kws}
                ax.plot(*segment, **kws)
        elif inner.startswith('box'):
            stats = mpl.cbook.boxplot_stats(obs)[0]
            pos = np.array(pos_dict[self.orient])
            if split:
                pos += (-1 if right_side else 1) * pos_dict['width'] / 2
            pos = ([pos, pos], [pos, pos], [pos])
            val = ([stats['whislo'], stats['whishi']], [stats['q1'], stats['q3']], [stats['med']])
            if self.orient == 'x':
                (x0, x1, x2), (y0, y1, y2) = (pos, val)
            else:
                (x0, x1, x2), (y0, y1, y2) = (val, pos)
            if split:
                offset = (1 if right_side else -1) * box_width / 72 / 2
                dx, dy = (offset, 0) if self.orient == 'x' else (0, -offset)
                trans = ax.transData + mpl.transforms.ScaledTranslation(dx, dy, ax.figure.dpi_scale_trans)
            else:
                trans = ax.transData
            line_kws = {'color': linecolor, 'transform': trans, **inner_kws, 'linewidth': whis_width}
            ax.plot(invx(x0), invy(y0), **line_kws)
            line_kws['linewidth'] = box_width
            ax.plot(invx(x1), invy(y1), **line_kws)
            dot_kws = {'marker': marker, 'markersize': box_width / 1.2, 'markeredgewidth': box_width / 5, 'transform': trans, **inner_kws, 'markeredgecolor': 'w', 'markerfacecolor': 'w', 'color': linecolor}
            ax.plot(invx(x2), invy(y2), **dot_kws)
    legend_artist = _get_patch_legend_artist(fill)
    common_kws = {**plot_kws, 'linewidth': linewidth, 'edgecolor': linecolor}
    self._configure_legend(ax, legend_artist, common_kws)