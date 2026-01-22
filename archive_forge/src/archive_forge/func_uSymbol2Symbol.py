from reportlab.graphics.shapes import Rect, Circle, Polygon, Drawing, Group
from reportlab.graphics.widgets.signsandsymbols import SmileyFace
from reportlab.graphics.widgetbase import Widget
from reportlab.lib.validators import isNumber, isColorOrNone, OneOf, Validator
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import isClass
from reportlab.graphics.widgets.flags import Flag, _Symbol
from math import sin, cos, pi
def uSymbol2Symbol(uSymbol, x, y, color):
    if isClass(uSymbol) and issubclass(uSymbol, Widget):
        size = 10.0
        symbol = uSymbol()
        symbol.x = x - size / 2
        symbol.y = y - size / 2
        try:
            symbol.size = size
            symbol.color = color
        except:
            pass
    elif isinstance(uSymbol, Marker) or isinstance(uSymbol, _Symbol):
        symbol = uSymbol.clone()
        if isinstance(uSymbol, Marker):
            symbol.fillColor = symbol.fillColor or color
        symbol.x, symbol.y = (x, y)
    elif callable(uSymbol):
        symbol = uSymbol(x, y, 5, color)
    else:
        symbol = None
    return symbol