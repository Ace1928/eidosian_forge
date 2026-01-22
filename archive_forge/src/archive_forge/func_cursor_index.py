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
def cursor_index(self, cursor=None):
    """Return the cursor index in the text/value.
        """
    if not cursor:
        cursor = self.cursor
    try:
        lines = self._lines
        if not lines:
            return 0
        flags = self._lines_flags
        index, cursor_row = cursor
        for _, line, flag in zip(range(min(cursor_row, len(lines))), lines, flags):
            index += len(line)
            if flag & FL_IS_LINEBREAK:
                index += 1
        if flags[cursor_row] & FL_IS_LINEBREAK:
            index += 1
        return index
    except IndexError:
        return 0