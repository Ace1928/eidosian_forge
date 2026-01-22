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
def set_min(self, min):
    """
        Set the lower value of the slider to *min*.

        Parameters
        ----------
        min : float
        """
    self.set_val((min, self.val[1]))