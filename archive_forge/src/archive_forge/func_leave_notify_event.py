import functools
import io
import os
import matplotlib as mpl
from matplotlib import _api, backend_tools, cbook
from matplotlib.backend_bases import (
from gi.repository import Gio, GLib, Gtk, Gdk, GdkPixbuf
from . import _backend_gtk
from ._backend_gtk import (  # noqa: F401 # pylint: disable=W0611
def leave_notify_event(self, controller):
    LocationEvent('figure_leave_event', self, *self._mpl_coords(), modifiers=self._mpl_modifiers())._process()