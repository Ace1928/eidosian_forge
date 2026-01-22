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
def imgVRange(h, va, fontSize):
    """return bottom,top offsets relative to baseline(0)"""
    if va == 'baseline':
        iyo = 0
    elif va in ('text-top', 'top'):
        iyo = fontSize - h
    elif va == 'middle':
        iyo = fontSize - (1.2 * fontSize + h) * 0.5
    elif va in ('text-bottom', 'bottom'):
        iyo = fontSize - 1.2 * fontSize
    elif va == 'super':
        iyo = 0.5 * fontSize
    elif va == 'sub':
        iyo = -0.5 * fontSize
    elif hasattr(va, 'normalizedValue'):
        iyo = va.normalizedValue(fontSize)
    else:
        iyo = va
    return (iyo, iyo + h)