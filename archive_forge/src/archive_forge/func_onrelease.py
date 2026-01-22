from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def onrelease(self, event):
    if self.ignore(event):
        return
    if self.verts is not None:
        self.verts.append(self._get_data_coords(event))
        if len(self.verts) > 2:
            self.callback(self.verts)
        self.line.remove()
    self.verts = None
    self.disconnect_events()