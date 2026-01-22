from functools import partial
from kivy.animation import Animation
from kivy.compat import string_types
from kivy.config import Config
from kivy.clock import Clock
from kivy.factory import Factory
from kivy.uix.stencilview import StencilView
from kivy.metrics import dp
from kivy.effects.dampedscroll import DampedScrollEffect
from kivy.properties import NumericProperty, BooleanProperty, AliasProperty, \
from kivy.uix.behaviors import FocusBehavior
def on_effect_cls(self, instance, cls):
    if isinstance(cls, string_types):
        cls = Factory.get(cls)
    self.effect_x = cls(target_widget=self._viewport)
    self.effect_x.bind(scroll=self._update_effect_x)
    self.effect_y = cls(target_widget=self._viewport)
    self.effect_y.bind(scroll=self._update_effect_y)