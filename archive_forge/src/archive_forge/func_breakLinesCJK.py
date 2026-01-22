from string import whitespace
from operator import truth
from unicodedata import category
from reportlab.pdfbase.pdfmetrics import stringWidth, getAscentDescent
from reportlab.platypus.paraparser import ParaParser, _PCT, _num as _parser_num, _re_us_value
from reportlab.platypus.flowables import Flowable
from reportlab.lib.colors import Color
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.geomutils import normalizeTRBL
from reportlab.lib.textsplit import wordSplit, ALL_CANNOT_START
from reportlab.lib.styles import ParagraphStyle
from copy import deepcopy
from reportlab.lib.abag import ABag
from reportlab.rl_config import decimalSymbol, _FUZZ, paraFontSizeHeightOffset,\
from reportlab.lib.utils import _className, isBytes, isStr
from reportlab.lib.rl_accel import sameFrag
import re
from types import MethodType
def breakLinesCJK(self, maxWidths):
    """Initially, the dumbest possible wrapping algorithm.
        Cannot handle font variations."""
    if not isinstance(maxWidths, (list, tuple)):
        maxWidths = [maxWidths]
    style = self.style
    self.height = 0
    _handleBulletWidth(self.bulletText, style, maxWidths)
    frags = self.frags
    nFrags = len(frags)
    if nFrags == 1 and (not hasattr(frags[0], 'cbDefn')) and (not style.endDots):
        f = frags[0]
        if hasattr(self, 'blPara') and getattr(self, '_splitpara', 0):
            return f.clone(kind=0, lines=self.blPara.lines)
        lines = []
        lineno = 0
        if hasattr(f, 'text'):
            text = f.text
        else:
            text = ''.join(getattr(f, 'words', []))
        lines = wordSplit(text, maxWidths, f.fontName, f.fontSize)
        wrappedLines = [(sp, [line]) for sp, line in lines]
        return f.clone(kind=0, lines=wrappedLines, ascent=f.fontSize, descent=-0.2 * f.fontSize)
    elif nFrags <= 0:
        return ParaLines(kind=0, fontSize=style.fontSize, fontName=style.fontName, textColor=style.textColor, lines=[], ascent=style.fontSize, descent=-0.2 * style.fontSize)
    if hasattr(self, 'blPara') and getattr(self, '_splitpara', 0):
        return self.blPara
    autoLeading = getattr(self, 'autoLeading', getattr(style, 'autoLeading', ''))
    calcBounds = autoLeading not in ('', 'off')
    return cjkFragSplit(frags, maxWidths, calcBounds)