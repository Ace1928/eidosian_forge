import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception(0)
def do_get_n_columns(self):
    """Internal method."""
    return self.on_get_n_columns()