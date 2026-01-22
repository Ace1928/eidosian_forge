import numpy as np
import param
from bokeh.models.glyphs import AnnularWedge
from ...core.data import GridInterface
from ...core.spaces import HoloMap
from ...core.util import dimension_sanitizer, is_nan
from .element import ColorbarPlot, CompositeElementPlot
from .selection import BokehOverlaySelectionDisplay
from .styles import base_properties, fill_properties, line_properties, text_properties
@classmethod
def is_radial(cls, heatmap):
    heatmap = heatmap.last if isinstance(heatmap, HoloMap) else heatmap
    opts = cls.lookup_options(heatmap, 'plot').options
    return any((o in opts for o in ('start_angle', 'radius_inner', 'radius_outer'))) and (not opts.get('radial') == False) or opts.get('radial', False)