from ctypes import *
from .base import FontException
import pyglet.lib
def float_to_f26p6(value):
    return int(value * (1 << 6))