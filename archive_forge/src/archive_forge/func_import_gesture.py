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
def import_gesture(self, data=None, filename=None, **kwargs):
    """Import a list of gestures as formatted by :meth:`export_gesture`.
        One of `data` or `filename` must be specified.

        This method accepts optional :meth:`Recognizer.filter` arguments,
        if none are specified then all gestures in specified data are
        imported."""
    if filename is not None:
        with open(filename, 'rb') as infile:
            data = infile.read()
    elif data is None:
        raise MultistrokeError('import_gesture needs data= or filename=')
    new = self.filter(db=self.parse_gesture(data), **kwargs)
    if new:
        self.db.extend(new)