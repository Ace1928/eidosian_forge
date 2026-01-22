from ctypes import *
from .base import FontException
import pyglet.lib
@classmethod
def check_and_raise_on_error(cls, errcode):
    if errcode != 0:
        raise cls(None, errcode)