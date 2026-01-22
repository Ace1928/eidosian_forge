import pickle
import base64
import zlib
from re import match as re_match
from collections import deque
from math import sqrt, pi, radians, acos, atan, atan2, pow, floor
from math import sin as math_sin, cos as math_cos
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.event import EventDispatcher
from kivy.properties import ListProperty
from kivy.compat import PY2
from io import BytesIO
def bounding_box(points):
    minx = float('infinity')
    miny = float('infinity')
    maxx = float('-infinity')
    maxy = float('-infinity')
    for px, py in points:
        if px < minx:
            minx = px
        if px > maxx:
            maxx = px
        if py < miny:
            miny = py
        if py > maxy:
            maxy = py
    return (minx, miny, maxx - minx + 1, maxy - miny + 1)