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
def dumpParagraphLines(P):
    print('dumpParagraphLines(<Paragraph @ %d>)' % id(P))
    lines = P.blPara.lines
    outw = sys.stdout.write
    for l, line in enumerate(lines):
        line = lines[l]
        if hasattr(line, 'words'):
            words = line.words
        else:
            words = line[1]
        nwords = len(words)
        outw('line%d: %d(%s)\n  ' % (l, nwords, str(getattr(line, 'wordCount', 'Unknown'))))
        for w in range(nwords):
            outw(" %d:'%s'" % (w, getattr(words[w], 'text', words[w])))
        print()