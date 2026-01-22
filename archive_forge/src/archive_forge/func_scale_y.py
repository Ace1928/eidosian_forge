import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
@scale_y.setter
def scale_y(self, scale_y):
    self._scale_y = scale_y
    self._vertex_list.scale[:] = (self._scale * self._scale_x, self._scale * scale_y)