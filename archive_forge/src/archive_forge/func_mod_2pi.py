from reportlab.graphics.shapes import Drawing, Polygon, Line
from math import pi
def mod_2pi(radians):
    radians = radians % _2pi
    if radians < -1e-06:
        radians += _2pi
    return radians