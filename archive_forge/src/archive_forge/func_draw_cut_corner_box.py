from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def draw_cut_corner_box(point1, point2, corner=0.5, color=colors.lightgreen, border=None, **kwargs):
    """Draw a box with the corners cut off."""
    x1, y1 = point1
    x2, y2 = point2
    if not corner:
        return draw_box(point1, point2, color, border)
    elif corner < 0:
        raise ValueError('Arrow head length ratio should be positive')
    strokecolor, color = _stroke_and_fill_colors(color, border)
    boxheight = y2 - y1
    boxwidth = x2 - x1
    x_corner = min(boxheight * 0.5 * corner, boxwidth * 0.5)
    y_corner = min(boxheight * 0.5 * corner, boxheight * 0.5)
    points = [x1, y1 + y_corner, x1, y2 - y_corner, x1 + x_corner, y2, x2 - x_corner, y2, x2, y2 - y_corner, x2, y1 + y_corner, x2 - x_corner, y1, x1 + x_corner, y1]
    return Polygon(deduplicate(points), strokeColor=strokecolor, strokeWidth=1, strokeLineJoin=1, fillColor=color, **kwargs)