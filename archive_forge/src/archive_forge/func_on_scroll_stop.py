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
def on_scroll_stop(self, touch, check_children=True):
    self._touch = None
    if check_children:
        touch.push()
        touch.apply_transform_2d(self.to_local)
        if self.dispatch_children('on_scroll_stop', touch):
            touch.pop()
            return True
        touch.pop()
    if self._get_uid('svavoid') in touch.ud:
        return
    if self._get_uid() not in touch.ud:
        return False
    self._touch = None
    uid = self._get_uid()
    ud = touch.ud[uid]
    not_in_bar = not touch.ud.get('in_bar_x', False) and (not touch.ud.get('in_bar_y', False))
    if self.do_scroll_x and self.effect_x and not_in_bar:
        self.effect_x.stop(touch.x)
    if self.do_scroll_y and self.effect_y and not_in_bar:
        self.effect_y.stop(touch.y)
    if ud['mode'] == 'unknown':
        if not ud['user_stopped']:
            self.simulate_touch_down(touch)
        Clock.schedule_once(partial(self._do_touch_up, touch), 0.2)
    ev = self._update_effect_bounds_ev
    if ev is None:
        ev = self._update_effect_bounds_ev = Clock.create_trigger(self._update_effect_bounds)
    ev()
    if 'button' in touch.profile and touch.button.startswith('scroll'):
        return True
    return self._get_uid() in touch.ud