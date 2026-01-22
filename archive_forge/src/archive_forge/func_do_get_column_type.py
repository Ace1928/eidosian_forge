import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception(GObject.TYPE_INVALID)
def do_get_column_type(self, index):
    """Internal method."""
    return self.on_get_column_type(index)