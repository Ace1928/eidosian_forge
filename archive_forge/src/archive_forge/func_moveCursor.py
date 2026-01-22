from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def moveCursor(self, dx, dy):
    """Starts a new line at an offset dx,dy from the start of the
        current line. This does not move the cursor relative to the
        current position, and it changes the current offset of every
        future line drawn (i.e. if you next do a textLine() call, it
        will move the cursor to a position one line lower than the
        position specificied in this call.  """
    if self._code and self._code[-1][-3:] == ' Td':
        L = self._code[-1].split()
        if len(L) == 3:
            del self._code[-1]
        else:
            self._code[-1] = ''.join(L[:-4])
        lastDx = float(L[-3])
        lastDy = float(L[-2])
        dx += lastDx
        dy -= lastDy
        self._x0 -= lastDx
        self._y0 -= lastDy
    self._code.append('%s Td' % fp_str(dx, -dy))
    self._x0 += dx
    self._y0 += dy
    self._x = self._x0
    self._y = self._y0