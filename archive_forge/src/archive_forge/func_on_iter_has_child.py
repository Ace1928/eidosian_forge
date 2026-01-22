import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_has_child(self, node):
    """Overridable.

        :Returns:
            True if the given node has children.
        """
    raise NotImplementedError