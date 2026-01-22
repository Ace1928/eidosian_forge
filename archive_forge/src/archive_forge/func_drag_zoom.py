from collections import namedtuple
from contextlib import ExitStack, contextmanager, nullcontext
from enum import Enum, IntEnum
import functools
import importlib
import inspect
import io
import itertools
import logging
import os
import sys
import time
import weakref
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib import (
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_managers import ToolManager
from matplotlib.cbook import _setattr_cm
from matplotlib.layout_engine import ConstrainedLayoutEngine
from matplotlib.path import Path
from matplotlib.texmanager import TexManager
from matplotlib.transforms import Affine2D
from matplotlib._enums import JoinStyle, CapStyle
def drag_zoom(self, event):
    """Callback for dragging in zoom mode."""
    start_xy = self._zoom_info.start_xy
    ax = self._zoom_info.axes[0]
    (x1, y1), (x2, y2) = np.clip([start_xy, [event.x, event.y]], ax.bbox.min, ax.bbox.max)
    key = event.key
    if self._zoom_info.cbar == 'horizontal':
        key = 'x'
    elif self._zoom_info.cbar == 'vertical':
        key = 'y'
    if key == 'x':
        y1, y2 = ax.bbox.intervaly
    elif key == 'y':
        x1, x2 = ax.bbox.intervalx
    self.draw_rubberband(event, x1, y1, x2, y2)