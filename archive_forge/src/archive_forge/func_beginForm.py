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
def beginForm(self, name, lowerx=0, lowery=0, upperx=None, uppery=None):
    """declare the current graphics stream to be a named form.
           A graphics stream can either be a page or a form, not both.
           Some operations (like bookmarking) are permitted for pages
           but not forms.  The form will not automatically be shown in the
           document but must be explicitly referenced using doForm in pages
           that require the form."""
    self.push_state_stack()
    self.init_graphics_state()
    if self._code or self._formData:
        self._pushAccumulators()
    self._formData = (name, lowerx, lowery, upperx, uppery)
    self._doc.inForm()