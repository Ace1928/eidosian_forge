from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def draw_polygon(list_of_points, color=colors.lightgreen, border=None, colour=None, **kwargs):
    """Draw polygon.

    Arguments:
     - list_of_point - list of (x,y) tuples for the corner coordinates
     - color / colour - The color for the box

    Returns a closed path object, beginning at (x1,y1) going round
    the four points in order, and filling with the passed colour.

    """
    if colour is not None:
        color = colour
        del colour
    strokecolor, color = _stroke_and_fill_colors(color, border)
    xy_list = []
    for x, y in list_of_points:
        xy_list.append(x)
        xy_list.append(y)
    return Polygon(deduplicate(xy_list), strokeColor=strokecolor, fillColor=color, strokewidth=0, **kwargs)