from collections import defaultdict
import numpy as np
import param
from bokeh.models import CategoricalColorMapper, CustomJS, FactorRange, Range1d, Whisker
from bokeh.models.tools import BoxSelectTool
from bokeh.transform import jitter
from ...core.data import Dataset
from ...core.dimension import dimension_name
from ...core.util import dimension_sanitizer, isfinite
from ...operation import interpolate_curve
from ...util.transform import dim
from ..mixins import AreaMixin, BarsMixin, SpikesMixin
from ..util import compute_sizes, get_min_distance
from .element import ColorbarPlot, ElementPlot, LegendPlot, OverlayPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import (
from .util import categorize_array
def get_batched_data(self, overlay, ranges):
    data = defaultdict(list)
    zorders = self._updated_zorders(overlay)
    for (key, el), zorder in zip(overlay.data.items(), zorders):
        el_opts = self.lookup_options(el, 'plot').options
        self.param.update(**{k: v for k, v in el_opts.items() if k not in OverlayPlot._propagate_options})
        style = self.lookup_options(el, 'style')
        style = style.max_cycles(len(self.ordering))[zorder]
        eldata, elmapping, style = self.get_data(el, ranges, style)
        if not eldata:
            continue
        for k, eld in eldata.items():
            data[k].append(eld)
        sdata, smapping = expand_batched_style(style, self._batched_style_opts, elmapping, nvals=1)
        elmapping.update(smapping)
        for k, v in sdata.items():
            data[k].append(v[0])
        for d, k in zip(overlay.kdims, key):
            sanitized = dimension_sanitizer(d.name)
            data[sanitized].append(k)
    data = {opt: vals for opt, vals in data.items() if not any((v is None for v in vals))}
    mapping = {{'x': 'xs', 'y': 'ys'}.get(k, k): v for k, v in elmapping.items()}
    return (data, mapping, style)