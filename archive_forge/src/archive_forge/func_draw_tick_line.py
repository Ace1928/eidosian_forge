import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def draw_tick_line(self, axis, color, radius, tick, labels_visible):
    tick_axis = {0: 1, 1: 0, 2: 1}[axis]
    tick_line = [[0, 0, 0], [0, 0, 0]]
    tick_line[0][axis] = tick_line[1][axis] = tick
    tick_line[0][tick_axis], tick_line[1][tick_axis] = (-radius, radius)
    self.draw_line(tick_line, color)
    if labels_visible:
        self.draw_tick_line_label(axis, color, radius, tick)