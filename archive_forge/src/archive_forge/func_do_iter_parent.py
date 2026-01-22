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
def do_iter_parent(self, child):
    """Internal method."""
    return self._create_tree_iter(self.on_iter_parent(self.get_user_data(child)))