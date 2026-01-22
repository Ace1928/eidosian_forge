from reportlab.lib.validators import isInt, isNumber, isString, isColorOrNone, isBoolean, EitherOr, isNumberOrNone
from reportlab.lib.attrmap import AttrMap, AttrMapValue
from reportlab.lib.colors import black
from reportlab.lib.utils import rl_exec
from reportlab.graphics.shapes import Rect, Group, String
from reportlab.graphics.charts.areas import PlotArea
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.barcode.widgets import BarcodeStandard93
def rect(self, x, y, w, h, **kw):
    for k, v in (('strokeColor', self.barStrokeColor), ('strokeWidth', self.barStrokeWidth), ('fillColor', self.barFillColor)):
        kw.setdefault(k, v)
    self._Gadd(Rect(self.x + x, self.y + y, w, h, **kw))