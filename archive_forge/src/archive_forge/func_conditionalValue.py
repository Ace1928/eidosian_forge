import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
@staticmethod
def conditionalValue(v, a):
    return a if v is NotSetOr._not_set else v