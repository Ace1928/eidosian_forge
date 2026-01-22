import uuid
import warnings
from ast import literal_eval
from collections import Counter, defaultdict
from functools import partial
from itertools import groupby, product
import numpy as np
import param
from panel.config import config
from panel.io.document import unlocked
from panel.io.notebook import push
from panel.io.state import state
from pyviz_comms import JupyterComm
from ..core import traversal, util
from ..core.data import Dataset, disable_pipeline
from ..core.element import Element, Element3D
from ..core.layout import Empty, Layout, NdLayout
from ..core.options import Compositor, SkipRendering, Store, lookup_options
from ..core.overlay import CompositeOverlay, NdOverlay, Overlay
from ..core.spaces import DynamicMap, HoloMap
from ..core.util import isfinite, stream_parameters
from ..element import Graph, Table
from ..selection import NoOpSelectionDisplay
from ..streams import RangeX, RangeXY, RangeY, Stream
from ..util.transform import dim
from .util import (
@property
def link_sources(self):
    """Returns potential Link or Stream sources."""
    if isinstance(self, GenericOverlayPlot):
        zorders = []
    elif self.batched:
        zorders = list(range(self.zorder, self.zorder + len(self.hmap.last)))
    else:
        zorders = [self.zorder]
    if isinstance(self, GenericOverlayPlot) and (not self.batched):
        sources = [self.hmap.last]
    elif not self.static or isinstance(self.hmap, DynamicMap):
        sources = [o for i, inputs in self.stream_sources.items() for o in inputs if i in zorders]
    else:
        sources = [self.hmap.last]
    return sources