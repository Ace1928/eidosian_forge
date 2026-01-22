from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required
def plot_cache(results, dsk, start_time, end_time, metric_name, palette='Viridis', label_size=60, **kwargs):
    """Visualize the results of profiling in a bokeh plot.

    Parameters
    ----------
    results : sequence
        Output of CacheProfiler.results
    dsk : dict
        The dask graph being profiled.
    start_time : float
        Start time of the profile in seconds
    end_time : float
        End time of the profile in seconds
    metric_name : string
        Metric used to measure cache size
    palette : string, optional
        Name of the bokeh palette to use, must be a member of
        bokeh.palettes.all_palettes.
    label_size: int (optional)
        Maximum size of output labels in plot, defaults to 60
    **kwargs
        Other keyword arguments, passed to bokeh.figure. These will override
        all defaults set by visualize.

    Returns
    -------
    The completed bokeh plot object.
    """
    bp = import_required('bokeh.plotting', _BOKEH_MISSING_MSG)
    from bokeh.models import HoverTool
    defaults = dict(title='Profile Results', tools='hover,save,reset,wheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use width instead of plot_width with Bokeh >= 3')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
        if BOKEH_VERSION().major >= 3:
            warnings.warn('Use height instead of plot_height with Bokeh >= 3')
    defaults.update(**kwargs)
    if results:
        starts, ends = list(zip(*results))[3:]
        tics = sorted(unique(starts + ends))
        groups = groupby(lambda d: pprint_task(d[1], dsk, label_size), results)
        data = {}
        for k, vals in groups.items():
            cnts = dict.fromkeys(tics, 0)
            for v in vals:
                cnts[v.cache_time] += v.metric
                cnts[v.free_time] -= v.metric
            data[k] = [0] + list(accumulate(add, pluck(1, sorted(cnts.items()))))
        tics = [0] + [i - start_time for i in tics]
        p = bp.figure(x_range=[0, end_time - start_time], **defaults)
        for (key, val), color in zip(data.items(), get_colors(palette, data.keys())):
            p.line('x', 'y', line_color=color, line_width=3, source=bp.ColumnDataSource({'x': tics, 'y': val, 'label': [key for i in val]}))
    else:
        p = bp.figure(y_range=[0, 10], x_range=[0, 10], **defaults)
    p.yaxis.axis_label = f'Cache Size ({metric_name})'
    p.xaxis.axis_label = 'Time (s)'
    hover = p.select(HoverTool)
    hover.tooltips = '\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Task:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@label</span>\n    </div>\n    '
    return p