from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def recolor_wheel(self):
    ppie = self._pieces_of_pie
    for idx, segment in enumerate(self.arcs):
        segment.change_color(sv=self.sv_s[int(self.sv_idx + idx / ppie)])