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
def process_key_on(self, touch):
    if not touch:
        return
    x, y = self.to_local(*touch.pos)
    key = self.get_key_at_pos(x, y)
    if not key:
        return
    key_data = key[0]
    displayed_char, internal, special_char, size = key_data
    line_nb, key_index = key[1]
    ud = touch.ud[self.uid] = {}
    ud['key'] = key
    uid = touch.uid
    if special_char is not None:
        if special_char in ('capslock', 'shift', 'layout', 'special'):
            if self._start_repeat_key_ev is not None:
                self._start_repeat_key_ev.cancel()
                self._start_repeat_key_ev = None
            self.repeat_touch = None
        if special_char == 'capslock':
            self.have_capslock = not self.have_capslock
            uid = -1
        elif special_char == 'shift':
            self.have_shift = True
        elif special_char == 'special':
            self.have_special = True
        elif special_char == 'layout':
            self.change_layout()
    b_keycode = special_char
    b_modifiers = self._get_modifiers()
    if self.get_parent_window().__class__.__module__ == 'kivy.core.window.window_sdl2' and internal:
        self.dispatch('on_textinput', internal)
    else:
        self.dispatch('on_key_down', b_keycode, internal, b_modifiers)
    self.active_keys[uid] = key[1]
    self.refresh_active_keys_layer()