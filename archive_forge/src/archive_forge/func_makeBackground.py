from reportlab.lib.validators import isNumber, isColorOrNone, isNoneOrShape
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics.shapes import Rect, Group, Line, Polygon
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import grey
def makeBackground(self):
    if self.background is not None:
        BG = self.background
        if isinstance(BG, Group):
            g = BG
            for bg in g.contents:
                bg.x = self.x
                bg.y = self.y
                bg.width = self.width
                bg.height = self.height
        else:
            g = Group()
            if type(BG) not in (type(()), type([])):
                BG = (BG,)
            for bg in BG:
                bg.x = self.x
                bg.y = self.y
                bg.width = self.width
                bg.height = self.height
                g.add(bg)
        return g
    else:
        strokeColor, strokeWidth, fillColor = (self.strokeColor, self.strokeWidth, self.fillColor)
        if strokeWidth and strokeColor or fillColor:
            g = Group()
            _3d_dy = getattr(self, '_3d_dy', None)
            x = self.x
            y = self.y
            h = self.height
            w = self.width
            if _3d_dy is not None:
                _3d_dx = self._3d_dx
                if fillColor and (not strokeColor):
                    from reportlab.lib.colors import Blacker
                    c = Blacker(fillColor, getattr(self, '_3d_blacken', 0.7))
                else:
                    c = strokeColor
                if not strokeWidth:
                    strokeWidth = 0.5
                if fillColor or strokeColor or c:
                    bg = Polygon([x, y, x, y + h, x + _3d_dx, y + h + _3d_dy, x + w + _3d_dx, y + h + _3d_dy, x + w + _3d_dx, y + _3d_dy, x + w, y], strokeColor=strokeColor or c or grey, strokeWidth=strokeWidth, fillColor=fillColor)
                    g.add(bg)
                    g.add(Line(x, y, x + _3d_dx, y + _3d_dy, strokeWidth=0.5, strokeColor=c))
                    g.add(Line(x + _3d_dx, y + _3d_dy, x + _3d_dx, y + h + _3d_dy, strokeWidth=0.5, strokeColor=c))
                    fc = Blacker(c, getattr(self, '_3d_blacken', 0.8))
                    g.add(Polygon([x, y, x + _3d_dx, y + _3d_dy, x + w + _3d_dx, y + _3d_dy, x + w, y], strokeColor=strokeColor or c or grey, strokeWidth=strokeWidth, fillColor=fc))
                    bg = Line(x + _3d_dx, y + _3d_dy, x + w + _3d_dx, y + _3d_dy, strokeWidth=0.5, strokeColor=c)
                else:
                    bg = None
            else:
                bg = Rect(x, y, w, h, strokeColor=strokeColor, strokeWidth=strokeWidth, fillColor=fillColor)
            if bg:
                g.add(bg)
            return g
        else:
            return None