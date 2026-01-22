import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def drawCentredString(self, x, y, text, mode=None, charSpace=0, direction=None, wordSpace=None):
    """Draws a string centred on the x coordinate. 
        
        We're British, dammit, and proud of our spelling!"""
    if not isinstance(text, str):
        text = text.decode('utf-8')
    width = self.stringWidth(text, self._fontname, self._fontsize)
    if charSpace:
        width += (len(text) - 1) * charSpace
    if wordSpace:
        width += (text.count(u' ') + text.count(u'\xa0') - 1) * wordSpace
    t = self.beginText(x - 0.5 * width, y, direction=direction)
    if mode is not None:
        t.setTextRenderMode(mode)
    if charSpace:
        t.setCharSpace(charSpace)
    if wordSpace:
        t.setWordSpace(wordSpace)
    t.textLine(text)
    if charSpace:
        t.setCharSpace(0)
    if wordSpace:
        t.setWordSpace(0)
    if mode is not None:
        t.setTextRenderMode(0)
    self.drawText(t)