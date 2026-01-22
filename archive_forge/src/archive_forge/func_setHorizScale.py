from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setHorizScale(self, horizScale):
    """Stretches text out horizontally"""
    self._horizScale = 100 + horizScale
    self._code.append('%s Tz' % fp_str(horizScale))