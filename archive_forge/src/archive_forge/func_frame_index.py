import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
@frame_index.setter
def frame_index(self, index):
    if self._animation is None:
        return
    self._frame_index = max(0, min(index, len(self._animation.frames) - 1))