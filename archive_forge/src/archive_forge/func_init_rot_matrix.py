import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def init_rot_matrix(self):
    pgl.glPushMatrix()
    pgl.glLoadIdentity()
    self._rot = get_model_matrix()
    pgl.glPopMatrix()