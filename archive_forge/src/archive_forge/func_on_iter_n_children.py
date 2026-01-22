import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_iter_n_children(self, node):
    """Overridable.

        :Returns:
            The number of children for the given node. If node is None,
            return the number of top level nodes.
        """
    raise NotImplementedError