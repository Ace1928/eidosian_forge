from math import cos, sin, pi, sqrt, atan
from colorsys import rgb_to_hsv, hsv_to_rgb
from kivy.clock import Clock
from kivy.graphics import Mesh, InstructionGroup, Color
from kivy.logger import Logger
from kivy.properties import (NumericProperty, BoundedNumericProperty,
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex, get_hex_from_color
def inertial_incr_sv_idx(self, dt):
    if self.sv_idx == len(self.sv_s) - self._piece_divisions:
        return False
    self.sv_idx += 1
    self.recolor_wheel()
    if dt * self._inertia_slowdown > self._inertia_cutoff:
        return False
    else:
        Clock.schedule_once(self.inertial_incr_sv_idx, dt * self._inertia_slowdown)