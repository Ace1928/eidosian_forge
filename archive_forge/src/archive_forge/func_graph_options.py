import re
import uuid
import numpy as np
import param
from ... import Tiles
from ...core import util
from ...core.element import Element
from ...core.spaces import DynamicMap
from ...streams import Stream
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import dim_range_key
from .plot import PlotlyPlot
from .util import (
def graph_options(self, element, ranges, style, is_geo=False, **kwargs):
    if self.overlay_dims:
        legend = ', '.join([d.pprint_value_string(v) for d, v in self.overlay_dims.items()])
    else:
        legend = element.label
    opts = dict(name=legend, **self.trace_kwargs(is_geo=is_geo))
    if self.trace_kwargs(is_geo=is_geo).get('type', None) in legend_trace_types:
        opts.update(showlegend=self.show_legend, legendgroup=element.group + '_' + legend)
    if self._style_key is not None:
        styles = self._apply_transforms(element, ranges, style)
        key_prefix_re = re.compile('^' + self._style_key + '_')
        styles = {key_prefix_re.sub('', k): v for k, v in styles.items()}
        opts[self._style_key] = {STYLE_ALIASES.get(k, k): v for k, v in styles.items()}
        for k in ['selectedpoints', 'visible']:
            if k in opts.get(self._style_key, {}):
                opts[k] = opts[self._style_key].pop(k)
    else:
        opts.update({STYLE_ALIASES.get(k, k): v for k, v in style.items() if k != 'cmap'})
    return opts