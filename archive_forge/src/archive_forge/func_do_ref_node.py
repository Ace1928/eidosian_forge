import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@handle_exception(None)
def do_ref_node(self, iter):
    self.on_ref_node(self.get_user_data(iter))