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
def project_poly(cb, msg):
    if not msg['data']:
        return msg
    projected = project_drawn(cb, msg)
    if projected is None:
        return msg
    split = projected.split()
    data = {d.name: [el.dimension_values(d) for el in split] for d in projected.dimensions()}
    xd, yd = projected.kdims
    data['xs'] = data.pop(xd.name)
    data['ys'] = data.pop(yd.name)
    return {'data': data}