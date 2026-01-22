import numpy as np
import holoviews as hv
from holoviews import opts
from . import get_aliases, all_original_names, palette, cm
from .sineramp import sineramp
def candy_buttons(name, cmap=None, size=450, **kwargs):
    if cmap is None:
        cmap = palette[name][:100]
        name = get_aliases(name)
    options = opts.Points(color='color', size=size / 13.0, tools=['hover'], yaxis=None, xaxis=None, height=size, width=size, cmap=cmap, **kwargs)
    return hv.Points(data, vdims='color').opts(options).relabel(name)