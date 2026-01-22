from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def makeDistancesList(list):
    """Returns a list of distances between adjacent numbers in some input list.

    E.g. [1, 1, 2, 3, 5, 7] -> [0, 1, 1, 2, 2]
    """
    d = []
    for i in range(len(list[:-1])):
        d.append(list[i + 1] - list[i])
    return d