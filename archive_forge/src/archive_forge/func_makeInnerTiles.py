from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isColorOrNone, isBoolean, isListOfNumbers, OneOf, isListOfColors, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.graphics.shapes import Drawing, Group, Line, Rect, LineShape, definePath, EmptyClipPath
from reportlab.graphics.widgetbase import Widget
def makeInnerTiles(self):
    group = Group()
    w, h = (self.width, self.height)
    if self.useRects == 1:
        cols = self.stripeColors
        if self.orientation == 'vertical':
            r = self.makeLinePosList(self.x, isX=1)
        elif self.orientation == 'horizontal':
            r = self.makeLinePosList(self.y, isX=0)
        dist = makeDistancesList(r)
        i = 0
        for j in range(len(dist)):
            if self.orientation == 'vertical':
                x = r[j]
                stripe = Rect(x, self.y, dist[j], h)
            elif self.orientation == 'horizontal':
                y = r[j]
                stripe = Rect(self.x, y, w, dist[j])
            stripe.fillColor = cols[i % len(cols)]
            stripe.strokeColor = None
            group.add(stripe)
            i = i + 1
    return group