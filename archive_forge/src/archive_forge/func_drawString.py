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
def drawString(self, x, y, text, mode=None, charSpace=0, direction=None, wordSpace=None):
    """Draws a string in the current text styles."""
    text = asUnicode(text)
    t = self.beginText(x, y, direction=direction)
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