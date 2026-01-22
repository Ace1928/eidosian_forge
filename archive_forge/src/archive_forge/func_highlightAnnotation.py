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
def highlightAnnotation(self, contents, Rect, QuadPoints=None, Color=[0.83, 0.89, 0.95], addtopage=1, name=None, relative=0, **kw):
    """
        Allows adding of a highlighted annotation.

        Rect: Mouseover area to show contents of annotation
        QuadPoints: List of four x/y points [TOP-LEFT, TOP-RIGHT, BOTTOM-LEFT, BOTTOM-RIGHT]
          These points outline the areas to highlight.
          You can have multiple groups of four to allow multiple highlighted areas.
          Is in the format [x1, y1, x2, y2, x3, y3, x4, y4, x1, y1, x2, y2, x3, y3, x4, y4] etc
          QuadPoints defaults to be area inside of passed in Rect
        Color: The color of the highlighting.
        """
    Rect = self._absRect(Rect, relative)
    if not QuadPoints:
        QuadPoints = pdfdoc.rect_to_quad(Rect)
    self._addAnnotation(pdfdoc.HighlightAnnotation(Rect, contents, QuadPoints, Color, **kw), name, addtopage)