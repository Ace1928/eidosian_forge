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
@verts.setter
def verts(self, xys):
    """
        Set the polygon vertices.

        This will remove any preexisting vertices, creating a complete polygon
        with the new vertices.
        """
    self._xys = [*xys, xys[0]]
    self._selection_completed = True
    self.set_visible(True)
    if self._draw_box and self._box is None:
        self._add_box()
    self._draw_polygon()