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
def do_iter_nth_child(self, parent, n):
    """Internal method."""
    if parent is None:
        data = self.on_iter_nth_child(None, n)
    else:
        data = self.on_iter_nth_child(self.get_user_data(parent), n)
    return self._create_tree_iter(data)