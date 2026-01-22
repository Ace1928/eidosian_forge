from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def stringLine(line, length):
    """simple case: line with just strings and spacings which can be ignored"""
    strings = []
    for x in line:
        if isinstance(x, str):
            strings.append(x)
    text = ' '.join(strings)
    result = [text, float(length)]
    nextlinemark = ('nextLine', 0)
    if line and line[-1] == nextlinemark:
        result.append(nextlinemark)
    return result