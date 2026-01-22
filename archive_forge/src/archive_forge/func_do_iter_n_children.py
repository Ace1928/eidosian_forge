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
def do_iter_n_children(self, iter):
    """Internal method."""
    if iter is None:
        return self.on_iter_n_children(None)
    return self.on_iter_n_children(self.get_user_data(iter))