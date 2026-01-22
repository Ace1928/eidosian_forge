from math import log10 as math_log10
from reportlab.lib.validators import    isNumber, isNumberOrNone, isListOfStringsOrNone, isListOfNumbers, \
from reportlab.lib.attrmap import *
from reportlab.lib import normalDate
from reportlab.graphics.shapes import Drawing, Line, PolyLine, Rect, Group, STATE_DEFAULTS, _textBoxLimits, _rotatedBoxLimits
from reportlab.graphics.widgetbase import Widget, TypedPropertyCollection
from reportlab.graphics.charts.textlabels import Label, PMVLabel, XLabel,  DirectDrawFlowable
from reportlab.graphics.charts.utils import nextRoundNumber
from reportlab.graphics.widgets.grids import ShadedRect
from reportlab.lib.colors import Color
from reportlab.lib.utils import isSeq
def timeLabelTextFormatter(self, val):
    u = self._unitd
    k = (u, tuple(self._tickValues))
    if k in self._saved_tickInfo:
        fmt = self._saved_tickInfo[k]
    else:
        uf = float(u)
        tv = [v / uf for v in self._tickValues]
        s = self._unit[0]
        if _allInt(tv):
            fmt = lambda x, uf=uf, s=s: '%.0f%s' % (x / uf, s)
        else:
            stv = ['%.10f' % v for v in tv]
            stvl = max((len(v.rstrip('0')) - v.index('.') - 1 for v in stv))
            if u == 1:
                fmt = lambda x, uf=uf, fmt='%%.%dfs' % stvl: fmt % (x / uf)
            else:
                fm = 24 if u == self._dc else 60
                fv = [(v - int(v)) * fm for v in tv]
                if _allInt(fv):
                    s1 = 'h' if u == self._dc else 'm' if u == self._mc else 's'
                    fmt = lambda x, uf=uf, fm=fm, fmt='%%d%s%%d%%s' % (s, s1): fmt % (int(x / uf), int((x / uf - int(x / uf)) * fm))
                else:
                    fmt = lambda x, uf=uf, fmt='%%.%df%s' % (stvl, s): fmt % (x / uf)
        self._saved_tickInfo[k] = fmt
    return fmt(val)