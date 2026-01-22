import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception((False, None))
def do_iter_children(self, parent):
    """Internal method."""
    data = self.get_user_data(parent) if parent else None
    return self._create_tree_iter(self.on_iter_children(data))