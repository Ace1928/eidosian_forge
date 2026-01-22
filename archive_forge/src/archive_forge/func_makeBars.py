import copy, functools
from ast import literal_eval
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isNumberOrNone, isColorOrNone, isString,\
from reportlab.lib.utils import isStr, yieldNoneSplits
from reportlab.graphics.widgets.markers import uSymbol2Symbol, isSymbol
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.graphics.widgetbase import TypedPropertyCollection, PropHolder, tpcGetItem
from reportlab.graphics.shapes import Line, Rect, Group, Drawing, PolyLine
from reportlab.graphics.charts.axes import XCategoryAxis, YValueAxis, YCategoryAxis, XValueAxis
from reportlab.graphics.charts.textlabels import BarChartLabel, NoneOrInstanceOfNA_Label
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.charts.legends import _objStr
from reportlab import cmp
def makeBars(self):
    from reportlab.graphics.charts.utils3d import _draw_3d_bar
    fg = _FakeGroup(cmp=self._cmpZ)
    self._makeBars(fg, fg)
    fg.sort()
    g = Group()
    theta_x = self.theta_x
    theta_y = self.theta_y
    fg_value = fg.value()
    cAStyle = self.categoryAxis.style
    if cAStyle == 'stacked':
        fg_value.reverse()
    elif cAStyle == 'mixed':
        fg_value = [_[1] for _ in sorted((((t[1], t[2], t[3], t[4]), t) for t in fg_value))]
    for t in fg_value:
        if t[0] == 0:
            z0, z1, x, y, width, height, rowNo, style = t[1:]
            dz = z1 - z0
            _draw_3d_bar(g, x, x + width, y, y + height, dz * theta_x, dz * theta_y, fillColor=style.fillColor, fillColorShaded=None, strokeColor=style.strokeColor, strokeWidth=style.strokeWidth, shading=0.45)
    for t in fg_value:
        if t == 1:
            z0, z1, x, y, width, height, rowNo, colNo = t[1:]
            BarChart._addBarLabel(self, g, rowNo, colNo, x, y, width, height)
    return g