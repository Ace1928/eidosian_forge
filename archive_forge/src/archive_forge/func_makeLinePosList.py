from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def makeLinePosList(self, start, isX=0):
    """Returns a list of positions where to place lines."""
    w, h = (self.width, self.height)
    if isX:
        length = w
    else:
        length = h
    if self.deltaSteps:
        r = [start + self.delta0]
        i = 0
        while 1:
            if r[-1] > start + length:
                del r[-1]
                break
            r.append(r[-1] + self.deltaSteps[i % len(self.deltaSteps)])
            i = i + 1
    else:
        r = frange(start + self.delta0, start + length, self.delta)
    r.append(start + length)
    if self.delta0 != 0:
        r.insert(0, start)
    return r