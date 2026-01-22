import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
def getValue(self, renderer, attr):
    return renderer.getStateValue(attr)