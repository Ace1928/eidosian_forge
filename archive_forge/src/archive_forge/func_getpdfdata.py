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
def getpdfdata(self):
    """Returns the PDF data that would normally be written to a file.
        If there is current data a ShowPage is executed automatically.
        After this operation the canvas must not be used further."""
    if len(self._code):
        self.showPage()
    s = self._doc.GetPDFData(self)
    if isUnicode(s):
        s = s.encode('utf-8')
    return s