import functools
import logging
import os
from pathlib import Path
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, GObject, Gtk, Gdk
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def size_allocate(self, widget, allocation):
    dpival = self.figure.dpi
    winch = allocation.width * self.device_pixel_ratio / dpival
    hinch = allocation.height * self.device_pixel_ratio / dpival
    self.figure.set_size_inches(winch, hinch, forward=False)
    ResizeEvent('resize_event', self)._process()
    self.draw_idle()