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
def bookmarkPage(self, key, fit='Fit', left=None, top=None, bottom=None, right=None, zoom=None):
    """
        This creates a bookmark to the current page which can
        be referred to with the given key elsewhere.

        PDF offers very fine grained control over how Acrobat
        reader is zoomed when people link to this. The default
        is to keep the user's current zoom settings. the last
        arguments may or may not be needed depending on the
        choice of 'fitType'.

        Fit types and the other arguments they use are:
        
        - XYZ left top zoom - fine grained control.  null
          or zero for any of the parameters means 'leave
          as is', so "0,0,0" will keep the reader's settings.
          NB. Adobe Reader appears to prefer "null" to 0's.

        - Fit - entire page fits in window

        - FitH top - top coord at top of window, width scaled
          to fit.

        - FitV left - left coord at left of window, height
          scaled to fit

        - FitR left bottom right top - scale window to fit
          the specified rectangle

        (question: do we support /FitB, FitBH and /FitBV
        which are hangovers from version 1.1 / Acrobat 3.0?)"""
    dest = self._bookmarkReference(key)
    self._doc.inPage()
    pageref = self._doc.thisPageRef()
    if left is None:
        left = 'null'
    if top is None:
        top = 'null'
    if bottom is None:
        bottom = 'null'
    if right is None:
        right = 'null'
    if zoom is None:
        zoom = 'null'
    if fit == 'XYZ':
        dest.xyz(left, top, zoom)
    elif fit == 'Fit':
        dest.fit()
    elif fit == 'FitH':
        dest.fith(top)
    elif fit == 'FitV':
        dest.fitv(left)
    elif fit == 'FitR':
        dest.fitr(left, bottom, right, top)
    elif fit == 'FitB':
        dest.fitb()
    elif fit == 'FitBH':
        dest.fitbh(top)
    elif fit == 'FitBV':
        dest.fitbv(left)
    else:
        raise ValueError('Unknown Fit type %s' % ascii(fit))
    dest.setPage(pageref)
    return dest