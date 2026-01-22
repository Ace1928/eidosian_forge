import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
def create_tree_iter(self, user_data):
    """Create a Gtk.TreeIter instance with the given user_data specific for this model.

        Use this method to create Gtk.TreeIter instance instead of directly calling
        Gtk.Treeiter(), this will ensure proper reference managment of wrapped used_data.
        """
    iter = Gtk.TreeIter()
    self.set_user_data(iter, user_data)
    return iter