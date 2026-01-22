from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required
def plot_resources(results, start_time, end_time, palette='Viridis', **kwargs):
    """Plot resource usage in a bokeh plot.

    Parameters
    ----------
    results : sequence
        Output of ResourceProfiler.results
    start_time : float
        Start time of the profile in seconds
    end_time : float
        End time of the profile in seconds
    palette : string, optional
        Name of the bokeh palette to use, must be a member of
        bokeh.palettes.all_palettes.
    **kwargs
        Other keyword arguments, passed to bokeh.figure. These will override
        all defaults set by plot_resources.

    Returns
    -------
    The completed bokeh plot object.
    """
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh import palettes
    from bokeh.models import LinearAxis, Range1d
    defaults = dict(title='Profile Results', tools='save,reset,xwheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use width instead of plot_width with Bokeh >= 3')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use height instead of plot_height with Bokeh >= 3')
    if 'label_size' in kwargs:
        kwargs.pop('label_size')
    defaults.update(**kwargs)
    if results:
        t, mem, cpu = zip(*results)
        left = start_time
        right = end_time
        t = [i - left for i in t]
        p = bp.figure(y_range=fix_bounds(0, max(cpu), 100), x_range=fix_bounds(0, right - left, 1), **defaults)
    else:
        t = mem = cpu = []
        p = bp.figure(y_range=(0, 100), x_range=(0, 1), **defaults)
    colors = palettes.all_palettes[palette][6]
    p.line(t, cpu, color=colors[0], line_width=4, legend_label='% CPU')
    p.yaxis.axis_label = '% CPU'
    p.extra_y_ranges = {'memory': Range1d(*fix_bounds(min(mem) if mem else 0, max(mem) if mem else 100, 100))}
    p.line(t, mem, color=colors[2], y_range_name='memory', line_width=4, legend_label='Memory')
    p.add_layout(LinearAxis(y_range_name='memory', axis_label='Memory (MB)'), 'right')
    p.xaxis.axis_label = 'Time (s)'
    return p