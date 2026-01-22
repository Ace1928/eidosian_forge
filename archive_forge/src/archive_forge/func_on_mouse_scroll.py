from pyglet.window import key
from pyglet.window.mouse import LEFT, RIGHT, MIDDLE
from sympy.plotting.pygletplot.util import get_direction_vectors, get_basis_vectors
def on_mouse_scroll(self, x, y, dx, dy):
    self.camera.zoom_relative([1, -1][self.invert_mouse_zoom] * dy, self.get_mouse_sensitivity())