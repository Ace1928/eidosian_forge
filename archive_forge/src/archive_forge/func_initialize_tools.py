import param
import cartopy.crs as ccrs
from holoviews.annotators import (
from holoviews.plotting.links import DataLink, VertexTableLink as hvVertexTableLink
from panel.util import param_name
from .element import Path
from .models.custom_tools import CheckpointTool, RestoreTool, ClearTool
from .links import VertexTableLink, PointTableLink, HvRectanglesTableLink, RectanglesTableLink
from .operation import project
from .streams import PolyVertexDraw, PolyVertexEdit
def initialize_tools(plot, element):
    """
    Initializes the Checkpoint and Restore tools.
    """
    cds = plot.handles['source']
    checkpoint = plot.state.select(type=CheckpointTool)
    restore = plot.state.select(type=RestoreTool)
    clear = plot.state.select(type=ClearTool)
    if checkpoint:
        checkpoint[0].sources.append(cds)
    if restore:
        restore[0].sources.append(cds)
    if clear:
        clear[0].sources.append(cds)