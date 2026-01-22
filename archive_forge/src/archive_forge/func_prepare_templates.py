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
def prepare_templates(self, **kwargs):
    """This method is used to prepare :class:`UnistrokeTemplate` objects
        within the gestures in self.db. This is useful if you want to minimize
        punishment of lazy resampling by preparing all vectors in advance. If
        you do this before a call to :meth:`Recognizer.export_gesture`, you
        will have the vectors computed when you load the data later.

        This method accepts optional :meth:`Recognizer.filter` arguments.

        `force_numpoints`, if specified, will prepare all templates to the
        given number of points (instead of each template's preferred n; ie
        :data:`UnistrokeTemplate.numpoints`). You normally don't want to
        do this."""
    for gesture in self.filter(**kwargs):
        for tpl in gesture:
            n = kwargs.get('force_numpoints', tpl.numpoints)
            tpl.prepare(n)