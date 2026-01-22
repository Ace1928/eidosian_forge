import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
def reset_bounding_box(self):
    self._bounding_box = [[None, None], [None, None], [None, None]]
    self._axis_ticks = [[], [], []]