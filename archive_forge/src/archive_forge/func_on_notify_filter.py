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
@functools.partial(dialog.connect, 'notify::filter')
def on_notify_filter(*args):
    name = dialog.get_filter().get_name()
    fmt = self.canvas.get_supported_filetypes_grouped()[name][0]
    dialog.set_current_name(str(Path(dialog.get_current_name()).with_suffix(f'.{fmt}')))