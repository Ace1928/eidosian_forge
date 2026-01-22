import warnings
from itertools import chain
from types import FunctionType
import bokeh
import bokeh.plotting
import numpy as np
import param
from bokeh.document.events import ModelChangedEvent
from bokeh.models import (
from bokeh.models.axes import CategoricalAxis, DatetimeAxis
from bokeh.models.formatters import (
from bokeh.models.layouts import TabPanel, Tabs
from bokeh.models.mappers import (
from bokeh.models.ranges import DataRange1d, FactorRange, Range1d
from bokeh.models.scales import LogScale
from bokeh.models.tickers import (
from bokeh.models.tools import Tool
from packaging.version import Version
from ...core import CompositeOverlay, Dataset, Dimension, DynamicMap, Element, util
from ...core.options import Keywords, SkipRendering, abbreviated_exception
from ...element import Annotation, Contours, Graph, Path, Tiles, VectorField
from ...streams import Buffer, PlotSize, RangeXY
from ...util.transform import dim
from ..plot import GenericElementPlot, GenericOverlayPlot
from ..util import color_intervals, dim_axis_label, dim_range_key, process_cmap
from .plot import BokehPlot
from .styles import (
from .tabular import TablePlot
from .util import (
def model_changed(self, model):
    """
        Determines if the bokeh model was just changed on the frontend.
        Useful to suppress boomeranging events, e.g. when the frontend
        just sent an update to the x_range this should not trigger an
        update on the backend.
        """
    callbacks = [cb for cbs in self.traverse(lambda x: x.callbacks) for cb in cbs]
    stream_metadata = [stream._metadata for cb in callbacks for stream in cb.streams if stream._metadata]
    return any((md['id'] == model.ref['id'] for models in stream_metadata for md in models.values()))