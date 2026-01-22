from time import perf_counter
import pyglet.gl as pgl
from sympy.plotting.pygletplot.managed_window import ManagedWindow
from sympy.plotting.pygletplot.plot_camera import PlotCamera
from sympy.plotting.pygletplot.plot_controller import PlotController
def update_caption(self, calc_verts_pos, calc_verts_len, calc_cverts_pos, calc_cverts_len):
    caption = self.title
    if calc_verts_len or calc_cverts_len:
        caption += ' (calculating'
        if calc_verts_len > 0:
            p = calc_verts_pos / calc_verts_len * 100
            caption += ' vertices %i%%' % p
        if calc_cverts_len > 0:
            p = calc_cverts_pos / calc_cverts_len * 100
            caption += ' colors %i%%' % p
        caption += ')'
    if self.caption != caption:
        self.set_caption(caption)