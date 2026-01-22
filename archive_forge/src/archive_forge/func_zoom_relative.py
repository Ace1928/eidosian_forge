import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def zoom_relative(self, clicks, sensitivity):
    if self.ortho:
        dist_d = clicks * sensitivity * 50.0
        min_dist = self.min_ortho_dist
        max_dist = self.max_ortho_dist
    else:
        dist_d = clicks * sensitivity
        min_dist = self.min_dist
        max_dist = self.max_dist
    new_dist = self._dist - dist_d
    if clicks < 0 and new_dist < max_dist or new_dist > min_dist:
        self._dist = new_dist