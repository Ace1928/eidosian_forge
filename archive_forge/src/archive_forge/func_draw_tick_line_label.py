import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def draw_tick_line_label(self, axis, color, radius, tick):
    if not self._p._label_axes:
        return
    tick_label_vector = [0, 0, 0]
    tick_label_vector[axis] = tick
    tick_label_vector[{0: 1, 1: 0, 2: 1}[axis]] = [-1, 1, 1][axis] * radius * 3.5
    self.draw_text(str(tick), tick_label_vector, color, scale=0.5)