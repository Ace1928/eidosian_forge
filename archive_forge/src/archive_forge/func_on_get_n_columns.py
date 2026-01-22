import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def on_get_n_columns(self):
    """Overridable.

        :Returns:
            The number of columns for this model.
        """
    raise NotImplementedError