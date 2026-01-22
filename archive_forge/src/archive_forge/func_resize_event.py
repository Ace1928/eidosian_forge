import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def resize_event(self, area, width, height):
    self._update_device_pixel_ratio()
    dpi = self.figure.dpi
    winch = width * self.device_pixel_ratio / dpi
    hinch = height * self.device_pixel_ratio / dpi
    self.figure.set_size_inches(winch, hinch, forward=False)
    ResizeEvent('resize_event', self)._process()
    self.draw_idle()