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
def update_from_scroll(self, *largs):
    """Force the reposition of the content, according to current value of
        :attr:`scroll_x` and :attr:`scroll_y`.

        This method is automatically called when one of the :attr:`scroll_x`,
        :attr:`scroll_y`, :attr:`pos` or :attr:`size` properties change, or
        if the size of the content changes.
        """
    if not self._viewport:
        self.g_translate.xy = self.pos
        return
    vp = self._viewport
    if vp.size_hint_x is not None:
        w = vp.size_hint_x * self.width
        if vp.size_hint_min_x is not None:
            w = max(w, vp.size_hint_min_x)
        if vp.size_hint_max_x is not None:
            w = min(w, vp.size_hint_max_x)
        vp.width = w
    if vp.size_hint_y is not None:
        h = vp.size_hint_y * self.height
        if vp.size_hint_min_y is not None:
            h = max(h, vp.size_hint_min_y)
        if vp.size_hint_max_y is not None:
            h = min(h, vp.size_hint_max_y)
        vp.height = h
    if vp.width > self.width or self.always_overscroll:
        sw = vp.width - self.width
        x = self.x - self.scroll_x * sw
    else:
        x = self.x
    if vp.height > self.height or self.always_overscroll:
        sh = vp.height - self.height
        y = self.y - self.scroll_y * sh
    else:
        y = self.top - vp.height
    vp.pos = (0, 0)
    self.g_translate.xy = (x, y)
    ev = self._bind_inactive_bar_color_ev
    if ev is None:
        ev = self._bind_inactive_bar_color_ev = Clock.create_trigger(self._bind_inactive_bar_color, 0.5)
    self.funbind('bar_inactive_color', self._change_bar_color)
    Animation.stop_all(self, '_bar_color')
    self.fbind('bar_color', self._change_bar_color)
    self._bar_color = self.bar_color
    ev()