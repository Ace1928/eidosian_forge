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
def configure_event(self, widget, event):
    if widget.get_property('window') is None:
        return
    w = event.width * self.device_pixel_ratio
    h = event.height * self.device_pixel_ratio
    if w < 3 or h < 3:
        return
    dpi = self.figure.dpi
    self.figure.set_size_inches(w / dpi, h / dpi, forward=False)
    return False