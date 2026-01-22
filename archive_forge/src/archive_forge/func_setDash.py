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
def setDash(self, array=[], phase=0):
    """Two notations.  pass two numbers, or an array and phase"""
    reason = ''
    if isinstance(array, (int, float)):
        array = (array, phase)
        phase = 0
    elif not isSeq(array):
        reason = 'array should be a sequence of numbers or a number'
    bad = [_ for _ in array if not isinstance(_, (int, float)) or _ < 0]
    if bad or not isinstance(phase, (int, float)) or phase < 0:
        reason = 'array & phase should be non-negative numbers'
    elif array and sum(array) <= 0:
        reason = 'dash cycle should be larger than zero'
    if reason:
        raise ValueError('setDash: array=%r phase=%r\n%s' % (array, phase, reason))
    self._code.append('[%s] %s d' % (fp_str(array), phase))