from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
@_docstring.Substitution(x0=0.5, y0=0.98, name='suptitle', ha='center', va='top', rc='title')
@_docstring.copy(_suplabels)
def suptitle(self, t, **kwargs):
    info = {'name': '_suptitle', 'x0': 0.5, 'y0': 0.98, 'ha': 'center', 'va': 'top', 'rotation': 0, 'size': 'figure.titlesize', 'weight': 'figure.titleweight'}
    return self._suplabels(t, info, **kwargs)