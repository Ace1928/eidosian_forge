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
def plot_boxes(self, width, dodge, gap, fill, whis, color, linecolor, linewidth, fliersize, plot_kws):
    iter_vars = ['hue']
    value_var = {'x': 'y', 'y': 'x'}[self.orient]

    def get_props(element, artist=mpl.lines.Line2D):
        return normalize_kwargs(plot_kws.pop(f'{element}props', {}), artist)
    if not fill and linewidth is None:
        linewidth = mpl.rcParams['lines.linewidth']
    bootstrap = plot_kws.pop('bootstrap', mpl.rcParams['boxplot.bootstrap'])
    plot_kws.setdefault('shownotches', plot_kws.pop('notch', False))
    box_artist = mpl.patches.Rectangle if fill else mpl.lines.Line2D
    props = {'box': get_props('box', box_artist), 'median': get_props('median'), 'whisker': get_props('whisker'), 'flier': get_props('flier'), 'cap': get_props('cap')}
    props['median'].setdefault('solid_capstyle', 'butt')
    props['whisker'].setdefault('solid_capstyle', 'butt')
    props['flier'].setdefault('markersize', fliersize)
    ax = self.ax
    for sub_vars, sub_data in self.iter_data(iter_vars, from_comp_data=True, allow_empty=False):
        ax = self._get_axes(sub_vars)
        grouped = sub_data.groupby(self.orient)[value_var]
        positions = sorted(sub_data[self.orient].unique().astype(float))
        value_data = [x.to_numpy() for _, x in grouped]
        stats = pd.DataFrame(mpl.cbook.boxplot_stats(value_data, whis=whis, bootstrap=bootstrap))
        orig_width = width * self._native_width
        data = pd.DataFrame({self.orient: positions, 'width': orig_width})
        if dodge:
            self._dodge(sub_vars, data)
        if gap:
            data['width'] *= 1 - gap
        capwidth = plot_kws.get('capwidths', 0.5 * data['width'])
        self._invert_scale(ax, data)
        _, inv = _get_transform_functions(ax, value_var)
        for stat in ['mean', 'med', 'q1', 'q3', 'cilo', 'cihi', 'whislo', 'whishi']:
            stats[stat] = inv(stats[stat])
        stats['fliers'] = stats['fliers'].map(inv)
        linear_orient_scale = getattr(ax, f'get_{self.orient}scale')() == 'linear'
        maincolor = self._hue_map(sub_vars['hue']) if 'hue' in sub_vars else color
        if fill:
            boxprops = {'facecolor': maincolor, 'edgecolor': linecolor, **props['box']}
            medianprops = {'color': linecolor, **props['median']}
            whiskerprops = {'color': linecolor, **props['whisker']}
            flierprops = {'markeredgecolor': linecolor, **props['flier']}
            capprops = {'color': linecolor, **props['cap']}
        else:
            boxprops = {'color': maincolor, **props['box']}
            medianprops = {'color': maincolor, **props['median']}
            whiskerprops = {'color': maincolor, **props['whisker']}
            flierprops = {'markeredgecolor': maincolor, **props['flier']}
            capprops = {'color': maincolor, **props['cap']}
        if linewidth is not None:
            for prop_dict in [boxprops, medianprops, whiskerprops, capprops]:
                prop_dict.setdefault('linewidth', linewidth)
        default_kws = dict(bxpstats=stats.to_dict('records'), positions=data[self.orient], widths=data['width'] if linear_orient_scale else 0, patch_artist=fill, vert=self.orient == 'x', manage_ticks=False, boxprops=boxprops, medianprops=medianprops, whiskerprops=whiskerprops, flierprops=flierprops, capprops=capprops, **{} if _version_predates(mpl, '3.6.0') else {'capwidths': capwidth})
        boxplot_kws = {**default_kws, **plot_kws}
        artists = ax.bxp(**boxplot_kws)
        ori_idx = ['x', 'y'].index(self.orient)
        if not linear_orient_scale:
            for i, box in enumerate(data.to_dict('records')):
                p0 = box['edge']
                p1 = box['edge'] + box['width']
                if artists['boxes']:
                    box_artist = artists['boxes'][i]
                    if fill:
                        box_verts = box_artist.get_path().vertices.T
                    else:
                        box_verts = box_artist.get_data()
                    box_verts[ori_idx][0] = p0
                    box_verts[ori_idx][3:] = p0
                    box_verts[ori_idx][1:3] = p1
                    if not fill:
                        box_artist.set_data(box_verts)
                    ax.update_datalim(np.transpose(box_verts), updatex=self.orient == 'x', updatey=self.orient == 'y')
                if artists['medians']:
                    verts = artists['medians'][i].get_xydata().T
                    verts[ori_idx][:] = (p0, p1)
                    artists['medians'][i].set_data(verts)
                if artists['caps']:
                    f_fwd, f_inv = _get_transform_functions(ax, self.orient)
                    for line in artists['caps'][2 * i:2 * i + 2]:
                        p0 = f_inv(f_fwd(box[self.orient]) - capwidth[i] / 2)
                        p1 = f_inv(f_fwd(box[self.orient]) + capwidth[i] / 2)
                        verts = line.get_xydata().T
                        verts[ori_idx][:] = (p0, p1)
                        line.set_data(verts)
        ax.add_container(BoxPlotContainer(artists))
    legend_artist = _get_patch_legend_artist(fill)
    self._configure_legend(ax, legend_artist, boxprops)