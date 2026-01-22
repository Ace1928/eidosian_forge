from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.rl_accel import fp_str
from reportlab.platypus.flowables import Flowable
from reportlab.lib import colors
from reportlab.lib.styles import _baseFontName
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.colors import black
def justifyAlign(self, line, lineLength, maxLength):
    diff = maxLength - lineLength
    spacecount = 0
    visible = 0
    first = 1
    for e in line:
        if isinstance(e, float) and e > TOOSMALLSPACE and visible:
            spacecount += 1
        elif first and (isinstance(e, str) or hasattr(e, 'width')):
            visible = 1
            first = 0
    if spacecount < 1:
        return line
    shift = diff / float(spacecount)
    if shift <= TOOSMALLSPACE:
        return line
    first = 1
    visible = 0
    result = []
    cursor = 0
    nline = len(line)
    while cursor < nline:
        e = line[cursor]
        result.append(e)
        if first and (isinstance(e, str) or hasattr(e, 'width')):
            visible = 1
        elif isinstance(e, float) and e > TOOSMALLSPACE and visible:
            expanded = e + shift
            result[-1] = expanded
        cursor += 1
    return result