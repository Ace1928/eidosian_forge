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
def release_zoom(self, event):
    """Callback for mouse button release in zoom to rect mode."""
    if self._zoom_info is None:
        return
    self.canvas.mpl_disconnect(self._zoom_info.cid)
    self.remove_rubberband()
    start_x, start_y = self._zoom_info.start_xy
    key = event.key
    if self._zoom_info.cbar == 'horizontal':
        key = 'x'
    elif self._zoom_info.cbar == 'vertical':
        key = 'y'
    if abs(event.x - start_x) < 5 and key != 'y' or (abs(event.y - start_y) < 5 and key != 'x'):
        self.canvas.draw_idle()
        self._zoom_info = None
        return
    for i, ax in enumerate(self._zoom_info.axes):
        twinx = any((ax.get_shared_x_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
        twiny = any((ax.get_shared_y_axes().joined(ax, prev) for prev in self._zoom_info.axes[:i]))
        ax._set_view_from_bbox((start_x, start_y, event.x, event.y), self._zoom_info.direction, key, twinx, twiny)
    self.canvas.draw_idle()
    self._zoom_info = None
    self.push_current()