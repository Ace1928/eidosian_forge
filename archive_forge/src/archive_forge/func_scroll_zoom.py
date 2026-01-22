import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
def scroll_zoom(self, event):
    if event.inaxes is None:
        return
    if event.button == 'up':
        scl = self.base_scale
    elif event.button == 'down':
        scl = 1 / self.base_scale
    else:
        scl = 1
    ax = event.inaxes
    ax._set_view_from_bbox([event.x, event.y, scl])
    if time.time() - self.lastscroll < self.scrollthresh:
        self.toolmanager.get_tool(_views_positions).back()
    self.figure.canvas.draw_idle()
    self.lastscroll = time.time()
    self.toolmanager.get_tool(_views_positions).push_current()