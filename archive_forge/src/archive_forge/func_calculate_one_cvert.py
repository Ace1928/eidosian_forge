import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase
def calculate_one_cvert(self, u, v):
    vert = self.verts[u][v]
    return self.color(vert[0], vert[1], vert[2], self.u_set[u], self.v_set[v])