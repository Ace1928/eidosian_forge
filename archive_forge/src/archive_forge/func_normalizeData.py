from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfStringsOrNone, OneOf,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Wedge
from reportlab.graphics.widgetbase import TypedPropertyCollection
from reportlab.graphics.charts.piecharts import AbstractPieChart, WedgeProperties, _addWedgeLabel, fixLabelOverlaps
from functools import reduce
def normalizeData(self, data=None):
    from operator import add
    sum = float(reduce(add, data, 0))
    return abs(sum) >= 1e-08 and list(map(lambda x, f=360.0 / sum: f * x, data)) or len(data) * [0]