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
def on_text_change(self, func):
    """
        When the text changes, call this *func* with event.

        A connection id is returned which can be used to disconnect.
        """
    return self._observers.connect('change', lambda text: func(text))