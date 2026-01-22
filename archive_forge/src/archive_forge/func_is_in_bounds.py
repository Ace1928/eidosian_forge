from reportlab.lib import pagesizes
from reportlab.lib import colors
from reportlab.graphics.shapes import Polygon
from math import pi, sin, cos
from itertools import islice
def is_in_bounds(self, value):
    """Check if given value is within the region selected for drawing.

        Arguments:
         - value - A base position

        """
    if value >= self.start and value <= self.end:
        return 1
    return 0