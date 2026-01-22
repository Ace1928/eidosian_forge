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
@staticmethod
def dumpFrags(frags, indent=4, full=False):
    R = ['[']
    aR = R.append
    for i, f in enumerate(frags):
        if full:
            aR('    [%r,' % f[0])
            for fx in f[1:]:
                aR('        (%s,)' % repr(fx[0]))
                aR('        %r),' % fx[1])
                aR('    ], #%d %s' % (i, f.__class__.__name__))
            aR('    ]')
        else:
            aR('[%r, %s], #%d %s' % (f[0], ', '.join(('(%s,%r)' % (fx[0].__class__.__name__, fx[1]) for fx in f[1:])), i, f.__class__.__name__))
    i = indent * ' '
    return i + ('\n' + i).join(R)