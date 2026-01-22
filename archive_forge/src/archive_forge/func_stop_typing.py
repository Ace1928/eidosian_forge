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
def stop_typing(self):
    if self.capturekeystrokes:
        self._on_stop_typing()
        self._on_stop_typing = None
        notifysubmit = True
    else:
        notifysubmit = False
    self.capturekeystrokes = False
    self.cursor.set_visible(False)
    self.ax.figure.canvas.draw()
    if notifysubmit and self.eventson:
        self._observers.process('submit', self.text)