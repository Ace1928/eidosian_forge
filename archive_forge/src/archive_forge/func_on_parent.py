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
def on_parent(self, instance, value):
    parent = self.textinput
    mode = self.mode
    if parent:
        self.content.clear_widgets()
        if mode == 'paste':
            self.but_selectall.opacity = 1
            widget_list = [self.but_selectall]
            if not parent.readonly:
                widget_list.append(self.but_paste)
        elif parent.readonly:
            widget_list = (self.but_copy,)
        else:
            widget_list = (self.but_cut, self.but_copy, self.but_paste)
        for widget in widget_list:
            self.content.add_widget(widget)