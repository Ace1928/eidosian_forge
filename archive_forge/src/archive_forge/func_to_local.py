from math import radians
from kivy.properties import BooleanProperty, AliasProperty, \
from kivy.vector import Vector
from kivy.uix.widget import Widget
from kivy.graphics.transformation import Matrix
def to_local(self, x, y, **k):
    p = self.transform_inv.transform_point(x, y, 0)
    return (p[0], p[1])