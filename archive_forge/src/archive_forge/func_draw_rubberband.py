import contextlib
import os
import signal
import socket
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macosx
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
def draw_rubberband(self, event, x0, y0, x1, y1):
    self.canvas.set_rubberband(int(x0), int(y0), int(x1), int(y1))