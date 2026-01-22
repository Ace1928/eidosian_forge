import pyglet.gl as pgl
from sympy.plotting.pygletplot.plot_rotation import get_spherical_rotatation
from sympy.plotting.pygletplot.util import get_model_matrix, model_to_screen, \
def set_rot_preset(self, preset_name):
    self.init_rot_matrix()
    try:
        r = self.rot_presets[preset_name]
    except AttributeError:
        raise ValueError('%s is not a valid rotation preset.' % preset_name)
    try:
        self.euler_rotate(r[0], 1, 0, 0)
        self.euler_rotate(r[1], 0, 1, 0)
        self.euler_rotate(r[2], 0, 0, 1)
    except AttributeError:
        pass