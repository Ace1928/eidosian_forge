import sys
import pyglet
from pyglet.gl import *
from pyglet import clock
from pyglet import event
from pyglet import graphics
from pyglet import image
def get_default_array_shader():
    return pyglet.gl.current_context.create_program((vertex_source, 'vertex'), (geometry_source, 'geometry'), (fragment_array_source, 'fragment'))