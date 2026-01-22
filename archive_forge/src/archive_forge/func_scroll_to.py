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
def scroll_to(self, widget, padding=10, animate=True):
    """Scrolls the viewport to ensure that the given widget is visible,
        optionally with padding and animation. If animate is True (the
        default), then the default animation parameters will be used.
        Otherwise, it should be a dict containing arguments to pass to
        :class:`~kivy.animation.Animation` constructor.

        .. versionadded:: 1.9.1
        """
    if not self.parent:
        return
    if hasattr(self._viewport, 'do_layout'):
        if self._viewport._trigger_layout.is_triggered:
            Clock.schedule_once(lambda *dt: self.scroll_to(widget, padding, animate))
            return
    if isinstance(padding, (int, float)):
        padding = (padding, padding)
    pos = self.parent.to_widget(*widget.to_window(*widget.pos))
    cor = self.parent.to_widget(*widget.to_window(widget.right, widget.top))
    dx = dy = 0
    if pos[1] < self.y:
        dy = self.y - pos[1] + dp(padding[1])
    elif cor[1] > self.top:
        dy = self.top - cor[1] - dp(padding[1])
    if pos[0] < self.x:
        dx = self.x - pos[0] + dp(padding[0])
    elif cor[0] > self.right:
        dx = self.right - cor[0] - dp(padding[0])
    dsx, dsy = self.convert_distance_to_scroll(dx, dy)
    sxp = min(1, max(0, self.scroll_x - dsx))
    syp = min(1, max(0, self.scroll_y - dsy))
    if animate:
        if animate is True:
            animate = {'d': 0.2, 't': 'out_quad'}
        Animation.stop_all(self, 'scroll_x', 'scroll_y')
        Animation(scroll_x=sxp, scroll_y=syp, **animate).start(self)
    else:
        self.scroll_x = sxp
        self.scroll_y = syp