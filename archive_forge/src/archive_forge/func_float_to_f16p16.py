from ctypes import *
from .base import FontException
import pyglet.lib
def float_to_f16p16(value):
    return int(value * (1 << 16))