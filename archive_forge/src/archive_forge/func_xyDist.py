from time import mktime, gmtime, strftime
from math import log10, pi, floor, sin, cos, hypot
import weakref
from reportlab.graphics.shapes import transformPoints, inverse, Ellipse, Group, String, numericXShift
from reportlab.lib.utils import flatten
from reportlab.pdfbase.pdfmetrics import stringWidth
def xyDist(xxx_todo_changeme, xxx_todo_changeme1):
    """return distance between two points"""
    x0, y0 = xxx_todo_changeme
    x1, y1 = xxx_todo_changeme1
    return hypot(x1 - x0, y1 - y0)