import sys
import warnings
from collections import UserList
import gi
from gi.repository import GObject
def gdk_window_get_geometry(window):
    return orig_gdk_window_get_geometry(window) + (window.get_visual().get_best_depth(),)