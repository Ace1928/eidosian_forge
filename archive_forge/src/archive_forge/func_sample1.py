from math import sin, cos, pi
from reportlab.lib import colors
from reportlab.lib.validators import isNumber, isListOfStringsOrNone, OneOf,\
from reportlab.lib.attrmap import *
from reportlab.graphics.shapes import Group, Drawing, Wedge
from reportlab.graphics.widgetbase import TypedPropertyCollection
from reportlab.graphics.charts.piecharts import AbstractPieChart, WedgeProperties, _addWedgeLabel, fixLabelOverlaps
from functools import reduce
def sample1():
    """Make up something from the individual Sectors"""
    d = Drawing(400, 400)
    g = Group()
    s1 = Wedge(centerx=200, centery=200, radius=150, startangledegrees=0, endangledegrees=120, radius1=100)
    s1.fillColor = colors.red
    s1.strokeColor = None
    d.add(s1)
    s2 = Wedge(centerx=200, centery=200, radius=150, startangledegrees=120, endangledegrees=240, radius1=100)
    s2.fillColor = colors.green
    s2.strokeColor = None
    d.add(s2)
    s3 = Wedge(centerx=200, centery=200, radius=150, startangledegrees=240, endangledegrees=260, radius1=100)
    s3.fillColor = colors.blue
    s3.strokeColor = None
    d.add(s3)
    s4 = Wedge(centerx=200, centery=200, radius=150, startangledegrees=260, endangledegrees=360, radius1=100)
    s4.fillColor = colors.gray
    s4.strokeColor = None
    d.add(s4)
    return d