from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def findNones(data):
    m = len(data)
    if None in data:
        b = 0
        while b < m and data[b] is None:
            b += 1
        if b == m:
            return data
        l = m - 1
        while data[l] is None:
            l -= 1
        l += 1
        if b or l:
            data = data[b:l]
        I = [i for i in range(len(data)) if data[i] is None]
        for i in I:
            data[i] = 0.5 * (data[i - 1] + data[i + 1])
        return (b, l, data)
    return (0, m, data)