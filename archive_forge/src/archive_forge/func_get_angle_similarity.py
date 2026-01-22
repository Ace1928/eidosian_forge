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
def get_angle_similarity(self, tpl, **kwargs):
    """(Internal use only) Compute the angle similarity between this
        Candidate and a UnistrokeTemplate object. Returns a number that
        represents the angle similarity (lower is more similar)."""
    n = kwargs.get('numpoints', self.numpoints)
    v1x, v1y = self.get_start_unit_vector(n, tpl.orientation_sens)
    v2x, v2y = tpl.get_start_unit_vector(n)
    n = v1x * v2x + v1y * v2y
    if n >= 1:
        return 0.0
    if n <= -1:
        return pi
    return acos(n)