from __future__ import annotations
import random
import warnings
from bisect import bisect_left
from itertools import cycle
from operator import add, itemgetter
from tlz import accumulate, groupby, pluck, unique
from dask.core import istask
from dask.utils import apply, funcname, import_required
def plot_tasks(results, dsk, start_time, end_time, palette='Viridis', label_size=60, **kwargs):
    """Visualize the results of profiling in a bokeh plot.

    Parameters
    ----------
    results : sequence
        Output of Profiler.results
    dsk : dict
        The dask graph being profiled.
    start_time : float
        Start time of the profile in seconds
    end_time : float
        End time of the profile in seconds
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
    defaults = dict(title='Profile Results', tools='hover,save,reset,xwheel_zoom,xpan', toolbar_location='above', width=800, height=300)
    if 'plot_width' in kwargs:
        kwargs['width'] = kwargs.pop('plot_width')
    if 'plot_height' in kwargs:
        kwargs['height'] = kwargs.pop('plot_height')
    defaults.update(**kwargs)
    if results:
        keys, tasks, starts, ends, ids = zip(*results)
        id_group = groupby(itemgetter(4), results)
        timings = {k: [i.end_time - i.start_time for i in v] for k, v in id_group.items()}
        id_lk = {t[0]: n for n, t in enumerate(sorted(timings.items(), key=itemgetter(1), reverse=True))}
        p = bp.figure(y_range=[str(i) for i in range(len(id_lk))], x_range=[0, end_time - start_time], **defaults)
        data = {}
        data['width'] = width = [e - s for s, e in zip(starts, ends)]
        data['x'] = [w / 2 + s - start_time for w, s in zip(width, starts)]
        data['y'] = [id_lk[i] + 1 for i in ids]
        data['function'] = funcs = [pprint_task(i, dsk, label_size) for i in tasks]
        data['color'] = get_colors(palette, funcs)
        data['key'] = [str(i) for i in keys]
        source = bp.ColumnDataSource(data=data)
        p.rect(source=source, x='x', y='y', height=1, width='width', color='color', line_color='gray')
    else:
        p = bp.figure(y_range=[str(i) for i in range(8)], x_range=[0, 10], **defaults)
    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.yaxis.axis_label = 'Worker ID'
    p.xaxis.axis_label = 'Time (s)'
    hover = p.select(HoverTool)
    hover.tooltips = '\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Key:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@key</span>\n    </div>\n    <div>\n        <span style="font-size: 14px; font-weight: bold;">Task:</span>&nbsp;\n        <span style="font-size: 10px; font-family: Monaco, monospace;">@function</span>\n    </div>\n    '
    hover.point_policy = 'follow_mouse'
    return p