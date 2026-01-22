import sys
import random
import collections
import ctypes
import platform
import warnings
import gi
from gi.repository import GObject
from gi.repository import Gtk
@classmethod
def from_iter(cls, iter):
    offset = sys.getsizeof(object())
    return ctypes.POINTER(cls).from_address(id(iter) + offset)