from reportlab.lib import colors
from reportlab.lib.validators import *
from reportlab.lib.attrmap import *
from reportlab.lib.utils import isStr, asUnicode
from reportlab.graphics import shapes
from reportlab.graphics.widgetbase import Widget
from reportlab.graphics import renderPDF
def demo(self):
    D = shapes.Drawing(200, 100)
    yn = YesNo()
    yn.x = 15
    yn.y = 25
    yn.size = 70
    yn.testValue = 0
    yn.draw()
    D.add(yn)
    yn2 = YesNo()
    yn2.x = 120
    yn2.y = 25
    yn2.size = 70
    yn2.testValue = 1
    yn2.draw()
    D.add(yn2)
    labelFontSize = 8
    D.add(shapes.String(yn.x + yn.size / 2, yn.y - 1.2 * labelFontSize, 'testValue=0', fillColor=colors.black, textAnchor='middle', fontSize=labelFontSize))
    D.add(shapes.String(yn2.x + yn2.size / 2, yn2.y - 1.2 * labelFontSize, 'testValue=1', fillColor=colors.black, textAnchor='middle', fontSize=labelFontSize))
    labelFontSize = 10
    D.add(shapes.String(yn.x + 85, yn.y - 20, self.__class__.__name__, fillColor=colors.black, textAnchor='middle', fontSize=labelFontSize))
    return D