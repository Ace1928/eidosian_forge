from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def get_gesture(self, touch):
    """Returns GestureContainer associated with given touch"""
    for g in self._gestures:
        if g.active and g.handles(touch):
            return g
    raise Exception('get_gesture() failed to identify ' + str(touch.uid))