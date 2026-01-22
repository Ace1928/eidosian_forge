from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def single_points_test(self):
    """Returns True if the gesture consists only of single-point strokes,
        we must discard it in this case, or an exception will be raised"""
    for tuid, l in self._strokes.items():
        if len(l.points) > 2:
            return False
    return True