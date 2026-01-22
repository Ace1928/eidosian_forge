from kivy import kivy_data_dir
from kivy.vector import Vector
from kivy.config import Config
from kivy.uix.scatter import Scatter
from kivy.uix.label import Label
from kivy.properties import ObjectProperty, NumericProperty, StringProperty, \
from kivy.logger import Logger
from kivy.graphics import Color, BorderImage, Canvas
from kivy.core.image import Image
from kivy.resources import resource_find
from kivy.clock import Clock
from io import open
from os.path import join, splitext, basename
from os import listdir
from json import loads
def collide_margin(self, x, y):
    """Do a collision test, and return True if the (x, y) is inside the
        vkeyboard margin.
        """
    mtop, mright, mbottom, mleft = self.margin_hint
    x_hint = x / self.width
    y_hint = y / self.height
    if x_hint > mleft and x_hint < 1.0 - mright and (y_hint > mbottom) and (y_hint < 1.0 - mtop):
        return False
    return True