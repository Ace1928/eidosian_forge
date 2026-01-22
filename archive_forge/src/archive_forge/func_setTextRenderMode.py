from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setTextRenderMode(self, mode):
    """Set the text rendering mode.

        0 = Fill text
        1 = Stroke text
        2 = Fill then stroke
        3 = Invisible
        4 = Fill text and add to clipping path
        5 = Stroke text and add to clipping path
        6 = Fill then stroke and add to clipping path
        7 = Add to clipping path

        after we start clipping we mustn't change the mode back until after the ET
        """
    assert mode in (0, 1, 2, 3, 4, 5, 6, 7), 'mode must be in (0,1,2,3,4,5,6,7)'
    if mode & 4 != self._clipping:
        mode |= 4
        self._clipping = mode & 4
    if self._textRenderMode != mode:
        self._textRenderMode = mode
        self._code.append('%d Tr' % mode)