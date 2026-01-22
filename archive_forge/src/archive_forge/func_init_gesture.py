from random import random
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.vector import Vector
from kivy.uix.floatlayout import FloatLayout
from kivy.graphics import Color, Line, Rectangle
from kivy.properties import (NumericProperty, BooleanProperty,
from colorsys import hsv_to_rgb
def init_gesture(self, touch):
    """Create a new gesture from touch, i.e. it's the first on
        surface, or was not close enough to any existing gesture (yet)"""
    col = self.color
    if self.use_random_color:
        col = hsv_to_rgb(random(), 1.0, 1.0)
    g = GestureContainer(touch, max_strokes=self.max_strokes, color=col)
    if self.draw_bbox:
        bb = g.bbox
        with self.canvas:
            Color(col[0], col[1], col[2], self.bbox_alpha, mode='rgba', group=g.id)
            g._bbrect = Rectangle(group=g.id, pos=(bb['minx'], bb['miny']), size=(bb['maxx'] - bb['minx'], bb['maxy'] - bb['miny']))
    self._gestures.append(g)
    return g