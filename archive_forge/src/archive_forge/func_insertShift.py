from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def insertShift(self, line, shift):
    result = []
    first = 1
    for e in line:
        if first and (isinstance(e, str) or hasattr(e, 'width')):
            result.append(shift)
            first = 0
        result.append(e)
    return result