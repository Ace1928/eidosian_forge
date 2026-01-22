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
def cjkFragSplit(frags, maxWidths, calcBounds, encoding='utf8'):
    """This attempts to be wordSplit for frags using the dumb algorithm"""
    U = []
    for f in frags:
        text = f.text
        if isBytes(text):
            text = text.decode(encoding)
        if text:
            U.extend([cjkU(t, f, encoding) for t in text])
        else:
            U.append(cjkU(text, f, encoding))
    lines = []
    i = widthUsed = lineStartPos = 0
    maxWidth = maxWidths[0]
    nU = len(U)
    while i < nU:
        u = U[i]
        i += 1
        w = u.width
        if hasattr(w, 'normalizedValue'):
            w._normalizer = maxWidth
            w = w.normalizedValue(maxWidth)
        widthUsed += w
        lineBreak = hasattr(u.frag, 'lineBreak')
        endLine = widthUsed > maxWidth + _FUZZ and widthUsed > 0 or lineBreak
        if endLine:
            extraSpace = maxWidth - widthUsed
            if not lineBreak:
                if ord(u) < 12288:
                    limitCheck = lineStartPos + i >> 1
                    for j in range(i - 1, limitCheck, -1):
                        uj = U[j]
                        if uj and category(uj) == 'Zs' or ord(uj) >= 12288:
                            k = j + 1
                            if k < i:
                                j = k + 1
                                extraSpace += sum((U[ii].width for ii in range(j, i)))
                                w = U[k].width
                                u = U[k]
                                i = j
                                break
                if u not in ALL_CANNOT_START and i > lineStartPos + 1:
                    i -= 1
                    extraSpace += w
            lines.append(makeCJKParaLine(U[lineStartPos:i], maxWidth, widthUsed, extraSpace, lineBreak, calcBounds))
            try:
                maxWidth = maxWidths[len(lines)]
            except IndexError:
                maxWidth = maxWidths[-1]
            lineStartPos = i
            widthUsed = 0
    if widthUsed > 0:
        lines.append(makeCJKParaLine(U[lineStartPos:], maxWidth, widthUsed, maxWidth - widthUsed, False, calcBounds))
    return ParaLines(kind=1, lines=lines)