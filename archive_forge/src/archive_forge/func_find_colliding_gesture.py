from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def find_colliding_gesture(self, touch):
    """Checks if a touch x/y collides with the bounding box of an existing
        gesture. If so, return it (otherwise returns None)
        """
    touch_x, touch_y = touch.pos
    for g in self._gestures:
        if g.active and (not g.handles(touch)) and g.accept_stroke():
            bb = g.bbox
            margin = self.bbox_margin
            minx = bb['minx'] - margin
            miny = bb['miny'] - margin
            maxx = bb['maxx'] + margin
            maxy = bb['maxy'] + margin
            if minx <= touch_x <= maxx and miny <= touch_y <= maxy:
                return g
    return None