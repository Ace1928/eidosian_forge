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
def splitLines0(frags, widths):
    """
    given a list of ParaFrags we return a list of ParaLines

    each ParaLine has
    1)  ExtraSpace
    2)  blankCount
    3)  [textDefns....]
    each text definition is a (ParaFrag, start, limit) triplet
    """
    lines = []
    lineNum = 0
    maxW = widths[lineNum]
    i = -1
    l = len(frags)
    lim = start = 0
    while 1:
        while i < l:
            while start < lim and text[start] == ' ':
                start += 1
            if start == lim:
                i += 1
                if i == l:
                    break
                start = 0
                f = frags[i]
                text = f.text
                lim = len(text)
            else:
                break
        if start == lim:
            break
        g = (None, None, None)
        line = []
        cLen = 0
        nSpaces = 0
        while cLen < maxW:
            j = text.find(' ', start)
            if j < 0:
                j == lim
            w = stringWidth(text[start:j], f.fontName, f.fontSize)
            cLen += w
            if cLen > maxW and line != []:
                cLen = cLen - w
                while g.text[lim] == ' ':
                    lim = lim - 1
                    nSpaces = nSpaces - 1
                break
            if j < 0:
                j = lim
            if g[0] is f:
                g[2] = j
            else:
                g = (f, start, j)
                line.append(g)
            if j == lim:
                i += 1