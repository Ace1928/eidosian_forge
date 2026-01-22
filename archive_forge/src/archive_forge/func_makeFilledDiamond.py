from reportlab.lib import colors
from reportlab.graphics.shapes import Rect, Circle, Polygon
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
def makeFilledDiamond(x, y, size, color):
    """Make a filled diamond marker."""
    d = size / 2.0
    poly = Polygon((x - d, y, x, y + d, x + d, y, x, y - d))
    poly.strokeColor = color
    poly.fillColor = color
    return poly