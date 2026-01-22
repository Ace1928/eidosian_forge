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
def drawAlignedString(self, x, y, text, pivotChar=rl_config.decimalSymbol, mode=None, charSpace=0, direction=None, wordSpace=None):
    """Draws a string aligned on the first '.' (or other pivot character).

        The centre position of the pivot character will be used as x.
        So, you could draw a straight line down through all the decimals in a
        column of numbers, and anything without a decimal should be
        optically aligned with those that have.

        There is one special rule to help with accounting formatting.  Here's
        how normal numbers should be aligned on the 'dot'. Look at the
        LAST two::
        
           12,345,67
              987.15
               42
           -1,234.56
             (456.78)
             (456)
               27 inches
               13cm
        
        Since the last three do not contain a dot, a crude dot-finding
        rule would place them wrong. So we test for the special case
        where no pivot is found, digits are present, but the last character
        is not a digit.  We then work back from the end of the string
        This case is a tad slower but hopefully rare.
        
        """
    parts = text.split(pivotChar, 1)
    pivW = self.stringWidth(pivotChar, self._fontname, self._fontsize)
    if len(parts) == 1 and digitPat.search(text) is not None and (text[-1] not in digits):
        leftText = parts[0][0:-1]
        rightText = parts[0][-1]
        while leftText[-1] not in digits:
            rightText = leftText[-1] + rightText
            leftText = leftText[0:-1]
        self.drawRightString(x - 0.5 * pivW, y, leftText, mode=mode, charSpace=charSpace, direction=direction, wordSpace=wordSpace)
        self.drawString(x - 0.5 * pivW, y, rightText, mode=mode, charSpace=charSpace, direction=direction, wordSpace=wordSpace)
    else:
        leftText = parts[0]
        self.drawRightString(x - 0.5 * pivW, y, leftText, mode=mode, charSpace=charSpace, direction=direction, wordSpace=wordSpace)
        if len(parts) > 1:
            rightText = pivotChar + parts[1]
            self.drawString(x - 0.5 * pivW, y, rightText, mode=mode, charSpace=charSpace, direction=direction, wordSpace=wordSpace)