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
def makeCJKParaLine(U, maxWidth, widthUsed, extraSpace, lineBreak, calcBounds):
    words = []
    CW = []
    f0 = FragLine()
    maxSize = maxAscent = minDescent = 0
    for u in U:
        f = u.frag
        fontSize = f.fontSize
        if calcBounds:
            cbDefn = getattr(f, 'cbDefn', None)
            if getattr(cbDefn, 'width', 0):
                descent, ascent = imgVRange(imgNormV(cbDefn.height, fontSize), cbDefn.valign, fontSize)
            else:
                ascent, descent = getAscentDescent(f.fontName, fontSize)
        else:
            ascent, descent = getAscentDescent(f.fontName, fontSize)
        maxSize = max(maxSize, fontSize)
        maxAscent = max(maxAscent, ascent)
        minDescent = min(minDescent, descent)
        if not sameFrag(f0, f):
            f0 = f0.clone()
            f0.text = u''.join(CW)
            words.append(f0)
            CW = []
            f0 = f
        CW.append(u)
    if CW:
        f0 = f0.clone()
        f0.text = u''.join(CW)
        words.append(f0)
    return FragLine(kind=1, extraSpace=extraSpace, wordCount=1, words=words[1:], fontSize=maxSize, ascent=maxAscent, descent=minDescent, maxWidth=maxWidth, currentWidth=widthUsed, lineBreak=lineBreak)