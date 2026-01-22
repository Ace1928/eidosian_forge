import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_children(self, parent):
    """Overridable.

        :Returns:
            The first child of parent or None if parent has no children.
            If parent is None, return the first node of the model.
        """
    raise NotImplementedError