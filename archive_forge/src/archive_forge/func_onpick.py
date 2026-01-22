import copy
from numbers import Integral, Number, Real
import logging
import numpy as np
import matplotlib as mpl
from . import _api, cbook, colors as mcolors, _docstring
from .artist import Artist, allow_rasterization
from .cbook import (
from .markers import MarkerStyle
from .path import Path
from .transforms import Bbox, BboxTransformTo, TransformedPath
from ._enums import JoinStyle, CapStyle
from . import _path
from .markers import (  # noqa
def onpick(self, event):
    """When the line is picked, update the set of selected indices."""
    if event.artist is not self.line:
        return
    self.ind ^= set(event.ind)
    ind = sorted(self.ind)
    xdata, ydata = self.line.get_data()
    self.process_selected(ind, xdata[ind], ydata[ind])