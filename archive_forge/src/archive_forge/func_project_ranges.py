import numpy as np
from pathlib import Path
from bokeh.models import CustomJS, CustomAction, PolyEditTool
from holoviews.core.ndmapping import UniformNdMapping
from holoviews.plotting.bokeh.callbacks import (
from holoviews.streams import (
from ...element.geo import _Element, Shape
from ...util import project_extents
from ...models import PolyVertexDrawTool, PolyVertexEditTool
from ...operation import project
from ...streams import PolyVertexEdit, PolyVertexDraw
from .plot import GeoOverlayPlot
def project_ranges(cb, msg, attributes):
    """
    Projects ranges supplied by a callback.
    """
    if skip(cb, msg, attributes):
        return msg
    plot = get_cb_plot(cb)
    x0, x1 = msg.get('x_range', (0, 1000))
    y0, y1 = msg.get('y_range', (0, 1000))
    extents = (x0, y0, x1, y1)
    x0, y0, x1, y1 = project_extents(extents, plot.projection, plot.current_frame.crs)
    coords = {'x_range': (x0, x1), 'y_range': (y0, y1)}
    return {k: v for k, v in coords.items() if k in attributes}