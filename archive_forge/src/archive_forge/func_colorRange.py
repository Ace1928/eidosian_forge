from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def colorRange(c0, c1, n):
    """Return a range of intermediate colors between c0 and c1"""
    if n == 1:
        return [c0]
    C = []
    if n > 1:
        lim = n - 1
        for i in range(n):
            C.append(colors.linearlyInterpolatedColor(c0, c1, 0, lim, i))
    return C