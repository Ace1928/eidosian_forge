import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def draw_axis_line_labels(self, axis, color, axis_line):
    if not self._p._label_axes:
        return
    axis_labels = [axis_line[0][:], axis_line[1][:]]
    axis_labels[0][axis] -= 0.3
    axis_labels[1][axis] += 0.3
    a_str = ['X', 'Y', 'Z'][axis]
    self.draw_text('-' + a_str, axis_labels[0], color)
    self.draw_text('+' + a_str, axis_labels[1], color)