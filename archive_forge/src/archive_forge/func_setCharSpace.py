from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setCharSpace(self, charSpace):
    """Adjusts inter-character spacing"""
    self._charSpace = charSpace
    self._code.append('%s Tc' % fp_str(charSpace))