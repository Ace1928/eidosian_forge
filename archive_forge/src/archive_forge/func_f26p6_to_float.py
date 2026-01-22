from ctypes import *
from .base import FontException
import pyglet.lib
def f26p6_to_float(value):
    return float(value) / (1 << 6)