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
def doForm(self, name):
    """use a form XObj in current operation stream.

        The form should either have been defined previously using
        beginForm ... endForm, or may be defined later.  If it is not
        defined at save time, an exception will be raised. The form
        will be drawn within the context of the current graphics
        state."""
    self._code.append('/%s Do' % self._doc.getXObjectName(name))
    self._formsinuse.append(name)