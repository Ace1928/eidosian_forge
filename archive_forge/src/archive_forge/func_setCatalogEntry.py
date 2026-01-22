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
def setCatalogEntry(self, key, value):
    from reportlab.pdfbase.pdfdoc import PDFDictionary, PDFArray, PDFString
    if isStr(value):
        value = PDFString(value)
    elif isinstance(value, (list, tuple)):
        value = PDFArray(value)
    elif isinstance(value, dict):
        value = PDFDictionary(value)
    setattr(self._doc.Catalog, key, value)