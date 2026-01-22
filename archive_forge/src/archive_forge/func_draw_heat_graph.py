from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_heat_graph(self, graph):
    """Return list of drawable elements for the heat graph.

        Arguments:
         - graph     Graph object

        """
    heat_elements = []
    data_quartiles = graph.quartiles()
    minval, maxval = (data_quartiles[0], data_quartiles[4])
    midval = (maxval + minval) / 2.0
    btm, ctr, top = self.track_radii[self.current_track_level]
    trackheight = top - btm
    start, end = self._current_track_start_end()
    data = intermediate_points(start, end, graph[start:end])
    for pos0, pos1, val in data:
        pos0angle, pos0cos, pos0sin = self.canvas_angle(pos0)
        pos1angle, pos1cos, pos1sin = self.canvas_angle(pos1)
        heat = colors.linearlyInterpolatedColor(graph.poscolor, graph.negcolor, maxval, minval, val)
        heat_elements.append(self._draw_arc(btm, top, pos0angle, pos1angle, heat, border=heat))
    return heat_elements