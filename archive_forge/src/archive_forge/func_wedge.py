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
def wedge(self, x1, y1, x2, y2, startAng, extent, stroke=1, fill=0):
    """Like arc, but connects to the centre of the ellipse.
        Most useful for pie charts and PacMan!"""
    p = pathobject.PDFPathObject(code=self._code)
    p.moveTo(0.5 * (x1 + x2), 0.5 * (y1 + y2))
    p.arcTo(x1, y1, x2, y2, startAng, extent)
    p.close()
    self._strokeAndFill(stroke, fill)