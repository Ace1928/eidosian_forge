from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def pairFixNones(pairs):
    Y = [x[1] for x in pairs]
    b, l, nY = findNones(Y)
    m = len(Y)
    if b or l < m or nY != Y:
        if b or l < m:
            pairs = pairs[b:l]
        pairs = [(x[0], y) for x, y in zip(pairs, nY)]
    return pairs