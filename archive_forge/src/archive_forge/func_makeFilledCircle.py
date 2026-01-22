from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Circle, Polygon
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def makeFilledCircle(x, y, size, color):
    """Make a hollow circle marker."""
    d = size / 2.0
    circle = Circle(x, y, d)
    circle.strokeColor = color
    circle.fillColor = color
    return circle