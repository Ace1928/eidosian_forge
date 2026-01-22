import re
import sys
import math
from os import environ
from weakref import ref
from itertools import chain, islice
from kivy.animation import Animation
from kivy.base import EventLoop
from kivy.cache import Cache
from kivy.clock import Clock
from kivy.config import Config
from kivy.core.window import Window
from kivy.metrics import inch
from kivy.utils import boundary, platform
from kivy.uix.behaviors import FocusBehavior
from kivy.core.text import Label, DEFAULT_FONT
from kivy.graphics import Color, Rectangle, PushMatrix, PopMatrix, Callback
from kivy.graphics.context_instructions import Transform
from kivy.graphics.texture import Texture
from kivy.uix.widget import Widget
from kivy.uix.bubble import Bubble
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.image import Image
from kivy.properties import StringProperty, NumericProperty, \
def scroll_text_from_swipe(self, touch):
    _scroll_timeout = (touch.time_update - touch.time_start) * 1000
    self._scroll_distance_x += abs(touch.dx)
    self._scroll_distance_y += abs(touch.dy)
    if not self._have_scrolled:
        if not (_scroll_timeout <= self.scroll_timeout and (self._scroll_distance_x >= self.scroll_distance or self._scroll_distance_y >= self.scroll_distance)):
            if _scroll_timeout <= self.scroll_timeout:
                return False
            else:
                self._enable_scroll = False
                self._cancel_update_selection(self._touch_down)
                return False
        self._have_scrolled = True
    self.cancel_long_touch_event()
    if self.multiline:
        max_scroll_y = max(0, self.minimum_height - self.height)
        self.scroll_y = min(max(0, self.scroll_y + touch.dy), max_scroll_y)
    else:
        max_scroll_x = self.get_max_scroll_x()
        self.scroll_x = min(max(0, self.scroll_x - touch.dx), max_scroll_x)
    self._trigger_update_graphics()
    self._position_handles()
    return True