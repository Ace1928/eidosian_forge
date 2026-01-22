from reportlab.graphics.shapes import Drawing, String, Group, Line, Circle, Polygon
from reportlab.lib import colors
from reportlab.graphics.shapes import ArcPath
from ._AbstractDrawer import AbstractDrawer, draw_polygon, intermediate_points
from ._AbstractDrawer import _stroke_and_fill_colors
from ._FeatureSet import FeatureSet
from ._GraphSet import GraphSet
from math import pi, cos, sin
def draw_test_tracks(self):
    """Draw blue test tracks with grene line down their center."""
    for track in self.drawn_tracks:
        btm, ctr, top = self.track_radii[track]
        self.drawing.add(Circle(self.xcenter, self.ycenter, top, strokeColor=colors.blue, fillColor=None))
        self.drawing.add(Circle(self.xcenter, self.ycenter, ctr, strokeColor=colors.green, fillColor=None))
        self.drawing.add(Circle(self.xcenter, self.ycenter, btm, strokeColor=colors.blue, fillColor=None))