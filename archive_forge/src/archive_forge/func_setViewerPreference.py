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
def setViewerPreference(self, pref, value):
    """set one of the allowed enbtries in the documents viewer preferences"""
    catalog = self._doc.Catalog
    VP = getattr(catalog, 'ViewerPreferences', None)
    if VP is None:
        from reportlab.pdfbase.pdfdoc import ViewerPreferencesPDFDictionary
        VP = catalog.ViewerPreferences = ViewerPreferencesPDFDictionary()
    VP[pref] = value