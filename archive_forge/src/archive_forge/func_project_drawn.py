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
def project_drawn(cb, msg):
    """
    Projects a drawn element to the declared coordinate system
    """
    stream = cb.streams[0]
    old_data = stream.data
    stream.update(data=msg['data'])
    element = stream.element
    stream.update(data=old_data)
    proj = cb.plot.projection
    if not isinstance(element, _Element) or element.crs == proj:
        return None
    crs = element.crs
    element.crs = proj
    return project(element, projection=crs)